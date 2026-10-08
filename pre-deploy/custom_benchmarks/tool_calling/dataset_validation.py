"""Validation helpers for the populated tool-calling benchmark dataset.

The migrated benchmark keeps symbolic ``{{steps[N].output}}`` references in
expected arguments.  This module validates the source representation without
rewriting those references or reducing a multi-hop trace to its first call.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .registry import TOOL_REGISTRY


_PLACEHOLDER_RE = re.compile(r"\{\{steps\[(\d+)\]\.output\}\}")
_RANGE_RE = re.compile(r"Range\s+(\d+)\s*-\s*(\d+)")
_PATTERN_RE = re.compile(r"Must match pattern\s+(.+?)\s*\([^)]*\)")


class DatasetValidationError(ValueError):
    """Raised when one or more benchmark scenarios violate the source contract."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        header = f"Dataset validation failed with {len(self.errors)} error(s):"
        return "\n".join([header, *(f"- {error}" for error in self.errors)])


def _error(errors: list[str], scenario_id: Any, location: str, message: str) -> None:
    errors.append(f"scenario {scenario_id!r}, {location}: {message}")


def _validate_argument_type(
    value: Any,
    parameter: Mapping[str, Any],
    errors: list[str],
    scenario_id: Any,
    location: str,
) -> None:
    declared_type = parameter.get("type")
    if declared_type == "string":
        if not isinstance(value, str):
            _error(errors, scenario_id, location, "must be a string")
    elif declared_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            _error(errors, scenario_id, location, "must be an integer")
    elif declared_type == "boolean":
        if not isinstance(value, bool):
            _error(errors, scenario_id, location, "must be a boolean")
    elif declared_type == "array of strings":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            _error(errors, scenario_id, location, "must be an array of strings")
    else:
        _error(errors, scenario_id, location, f"uses unsupported registry type {declared_type!r}")


def _validate_parameter_constraints(
    value: Any,
    parameter: Mapping[str, Any],
    errors: list[str],
    scenario_id: Any,
    location: str,
) -> None:
    constraints = str(parameter.get("constraints", ""))
    if isinstance(value, str) and "Non-empty" in constraints and not value.strip():
        _error(errors, scenario_id, location, "must not be empty")

    pattern_match = _PATTERN_RE.search(constraints)
    if pattern_match and isinstance(value, str):
        pattern = pattern_match.group(1)
        try:
            matches = re.fullmatch(pattern, value) is not None
        except re.error as exc:
            _error(errors, scenario_id, location, f"has invalid registry pattern: {exc}")
            matches = True
        if not matches:
            _error(errors, scenario_id, location, "does not satisfy the registry pattern")

    if "sections" in constraints and isinstance(value, str):
        required_sections = ("INTENT:", "COMPONENTS:", "CONSTRAINTS:", "CONTEXT:")
        missing = [section for section in required_sections if section not in value]
        if missing:
            _error(errors, scenario_id, location, f"is missing required sections: {', '.join(missing)}")

    range_match = _RANGE_RE.search(constraints)
    if range_match and isinstance(value, int) and not isinstance(value, bool):
        lower, upper = (int(part) for part in range_match.groups())
        if not lower <= value <= upper:
            _error(errors, scenario_id, location, f"must be between {lower} and {upper}")


def _validate_argument_object(
    call: Mapping[str, Any],
    tool_name: str,
    hop_index: int,
    tool_registry: Mapping[str, Mapping[str, Any]],
    errors: list[str],
    scenario_id: Any,
    location: str,
) -> None:
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        _error(errors, scenario_id, location, "arguments must be an object")
        return

    tool_definition = tool_registry[tool_name]
    parameters = [
        *tool_definition.get("required_parameters", []),
        *tool_definition.get("optional_parameters", []),
    ]
    parameter_by_name = {parameter["name"]: parameter for parameter in parameters}

    for argument_name in arguments:
        if argument_name not in parameter_by_name:
            _error(errors, scenario_id, f"{location}.arguments", f"unknown parameter {argument_name!r}")

    for parameter in tool_definition.get("required_parameters", []):
        name = parameter["name"]
        if name not in arguments:
            _error(errors, scenario_id, f"{location}.arguments", f"missing required parameter {name!r}")

    for name, value in arguments.items():
        parameter = parameter_by_name.get(name)
        if parameter is None:
            continue
        # Only an explicitly declared optional default of ``None`` permits a
        # null argument.  Required parameters without a default must still
        # satisfy their declared type.
        if value is None and "default" in parameter and parameter.get("default") is None:
            continue
        value_location = f"{location}.arguments[{name!r}]"
        _validate_argument_type(value, parameter, errors, scenario_id, value_location)
        _validate_parameter_constraints(value, parameter, errors, scenario_id, value_location)
        _validate_placeholder_references(value, hop_index, errors, scenario_id, value_location)


def _validate_placeholder_references(
    value: Any,
    hop_index: int,
    errors: list[str],
    scenario_id: Any,
    location: str,
) -> None:
    """Require every symbolic dependency to point at a strictly earlier hop."""
    if isinstance(value, dict):
        for key, nested_value in value.items():
            _validate_placeholder_references(
                nested_value, hop_index, errors, scenario_id, f"{location}.{key}"
            )
        return
    if isinstance(value, list):
        for index, nested_value in enumerate(value):
            _validate_placeholder_references(
                nested_value, hop_index, errors, scenario_id, f"{location}[{index}]"
            )
        return
    if not isinstance(value, str):
        return

    matches = list(_PLACEHOLDER_RE.finditer(value))
    for match in matches:
        referenced_hop = int(match.group(1))
        if referenced_hop >= hop_index:
            _error(
                errors,
                scenario_id,
                location,
                f"placeholder references step {referenced_hop}, but only steps before hop {hop_index} are valid",
            )

    # Catch malformed versions such as {{steps[0].result}} or {{steps[x].output}}
    # without rejecting ordinary prose that merely mentions the word "steps".
    if "{{steps[" in value:
        remainder = _PLACEHOLDER_RE.sub("", value)
        if "{{steps[" in remainder:
            _error(errors, scenario_id, location, "contains a malformed step output placeholder")


def validate_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    tool_registry: Mapping[str, Mapping[str, Any]] = TOOL_REGISTRY,
) -> list[Mapping[str, Any]]:
    """Validate complete ordered golden truth for every source scenario.

    The input is returned unchanged so callers preserve source metadata and
    symbolic placeholders.  All errors are accumulated before raising, which
    lets operators fix a dataset in one pass and ensures validation happens
    before any candidate execution is opened.
    """
    errors: list[str] = []
    if not isinstance(scenarios, Sequence) or isinstance(scenarios, (str, bytes)):
        raise DatasetValidationError(["dataset root must be a list of scenario objects"])

    seen_ids: set[str] = set()
    for scenario_index, scenario in enumerate(scenarios):
        if not isinstance(scenario, Mapping):
            errors.append(f"scenario index {scenario_index}: record must be an object")
            continue

        scenario_id_value = scenario.get("scenario_id")
        scenario_label = (
            scenario_id_value
            if isinstance(scenario_id_value, str) and scenario_id_value.strip()
            else f"<index {scenario_index}>"
        )
        if not isinstance(scenario_id_value, str) or not scenario_id_value.strip():
            _error(errors, scenario_label, "scenario_id", "must be a non-empty string")
        elif scenario_id_value in seen_ids:
            _error(errors, scenario_id_value, "scenario_id", "must be unique")
        else:
            seen_ids.add(scenario_id_value)

        question = scenario.get("question")
        if question is not None:
            if not isinstance(question, str) or not question.strip():
                _error(errors, scenario_label, "question", "must be a non-empty string")
        else:
            questions = scenario.get("questions")
            if not isinstance(questions, list) or not questions:
                _error(errors, scenario_label, "questions", "must contain at least one question")
            elif not isinstance(questions[0], str) or not questions[0].strip():
                _error(errors, scenario_label, "questions[0]", "must be a non-empty string")

        enabled_tools = scenario.get("enabled_tools")
        if not isinstance(enabled_tools, Mapping):
            _error(errors, scenario_label, "enabled_tools", "must be an object")
            enabled_tools = {}
        for tool_name, enabled in enabled_tools.items():
            if not isinstance(tool_name, str) or tool_name not in tool_registry:
                _error(errors, scenario_label, "enabled_tools", f"unknown registry tool {tool_name!r}")
            if not isinstance(enabled, bool):
                _error(errors, scenario_label, f"enabled_tools[{tool_name!r}]", "must be boolean")

        hop_count = scenario.get("hop_count")
        expected_tools = scenario.get("expected_tools")
        golden_truth = scenario.get("golden_truth")
        if isinstance(hop_count, bool) or not isinstance(hop_count, int) or hop_count < 0:
            _error(errors, scenario_label, "hop_count", "must be a non-negative integer")
            hop_count_valid = False
        else:
            hop_count_valid = True
        if not isinstance(expected_tools, list) or not all(isinstance(tool, str) for tool in expected_tools):
            _error(errors, scenario_label, "expected_tools", "must be a list of tool-name strings")
            expected_tools = []
        else:
            for expected_index, tool_name in enumerate(expected_tools):
                if tool_name not in tool_registry:
                    _error(
                        errors,
                        scenario_label,
                        f"expected_tools[{expected_index}]",
                        f"unknown registry tool {tool_name!r}",
                    )
                elif enabled_tools.get(tool_name) is not True:
                    _error(
                        errors,
                        scenario_label,
                        f"expected_tools[{expected_index}]",
                        "tool must be enabled for the scenario",
                    )
        if not isinstance(golden_truth, list):
            _error(errors, scenario_label, "golden_truth", "must be a list")
            golden_truth = []

        # Continue validating the remaining records after a malformed hop
        # count, but do not apply length/index rules that require an integer.
        if not hop_count_valid:
            continue

        explicit_null_truth = golden_truth == [{"tool_name": None, "arguments": {}}]
        if hop_count == 0:
            if expected_tools:
                _error(errors, scenario_label, "expected_tools", "must be empty for abstention")
            if golden_truth not in ([], [{"tool_name": None, "arguments": {}}]):
                _error(errors, scenario_label, "golden_truth", "must be empty or the explicit null-tool abstention object")
            if explicit_null_truth:
                continue
            continue

        if len(golden_truth) != hop_count:
            _error(errors, scenario_label, "golden_truth", f"length {len(golden_truth)} must equal hop_count {hop_count}")
        if len(expected_tools) != hop_count:
            _error(errors, scenario_label, "expected_tools", f"length {len(expected_tools)} must equal hop_count {hop_count}")
        if len(golden_truth) != len(expected_tools):
            _error(errors, scenario_label, "golden_truth", "must have one call for every expected tool")

        for hop_index, call in enumerate(golden_truth):
            location = f"golden_truth[{hop_index}]"
            if not isinstance(call, Mapping):
                _error(errors, scenario_label, location, "must be an object")
                continue
            tool_name = call.get("tool_name")
            if not isinstance(tool_name, str) or not tool_name:
                _error(errors, scenario_label, f"{location}.tool_name", "must be a non-empty tool name")
                continue
            if hop_index < len(expected_tools) and tool_name != expected_tools[hop_index]:
                _error(
                    errors,
                    scenario_label,
                    f"{location}.tool_name",
                    f"{tool_name!r} must equal expected_tools[{hop_index}] {expected_tools[hop_index]!r}",
                )
            if tool_name not in tool_registry:
                _error(errors, scenario_label, f"{location}.tool_name", f"unknown registry tool {tool_name!r}")
                continue
            if enabled_tools.get(tool_name) is not True:
                _error(errors, scenario_label, f"{location}.tool_name", "tool must be enabled for the scenario")
            _validate_argument_object(
                call,
                tool_name,
                hop_index,
                tool_registry,
                errors,
                scenario_label,
                location,
            )

    if errors:
        raise DatasetValidationError(errors)
    return list(scenarios)


def validate_dataset_file(
    dataset_path: Path | str,
    tool_registry: Mapping[str, Mapping[str, Any]] = TOOL_REGISTRY,
) -> list[Mapping[str, Any]]:
    """Load and validate a JSON dataset before any benchmark execution."""
    path = Path(dataset_path)
    try:
        scenarios = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError([f"could not load {path}: {exc}"]) from exc
    return validate_scenarios(scenarios, tool_registry=tool_registry)
