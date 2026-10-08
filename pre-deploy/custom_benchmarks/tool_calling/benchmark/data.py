"""Scenario loading, golden/eval dataset construction, and dry-run preview.

Owns everything that turns the source seed JSON into validated scenarios and
into the ``EvalDataset``/``GoldenTestSet`` structures the evaluator consumes,
plus the credential-free dry-run request preview.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from pre_deploy.input import EvalDataset, GoldenPair, GoldenTestSet, build_eval_dataset
from custom_benchmarks.tool_calling.agentic.config import redact_sensitive_data
from custom_benchmarks.tool_calling.agentic.runner import build_agentic_request
from custom_benchmarks.tool_calling.catalog import project_tool_catalog
from custom_benchmarks.tool_calling.dataset_validation import validate_scenarios
from custom_benchmarks.tool_calling.models import ToolCallObservation
from custom_benchmarks.tool_calling.serialization import (
    CANONICAL_TRACE_SCHEMA_VERSION,
    canonical_json,
    serialize_expected_trace,
    serialize_observed_trace,
)

# Statuses that mean a candidate never produced a completed trace.  These are
# excluded from the evaluator dataset so an empty trace cannot masquerade as a
# successful abstention.
_EXECUTION_FAILURE_STATUSES = frozenset({"transport_error", "error_eos", "incomplete"})


def _placeholder_dependencies(value: Any, path: str = "arguments") -> list[dict[str, Any]]:
    """Collect symbolic prior-step references without changing their values."""
    references: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            references.extend(_placeholder_dependencies(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            references.extend(_placeholder_dependencies(item, f"{path}[{index}]"))
    elif isinstance(value, str):
        for match in re.finditer(r"\{\{steps\[(\d+)\]\.output\}\}", value):
            references.append({"path": path, "step": int(match.group(1)), "reference": match.group(0)})
    return references


def _render_evaluator_tool_catalog(scenario: Mapping[str, Any]) -> str:
    """Render the enabled registry catalog for evaluator-only metadata.

    The evaluator's existing prompt API calls this context ``system_prompt``.
    It is deliberately populated with the local registry projection and is
    never passed to :func:`build_agentic_request` or the WebSocket client.
    """
    return canonical_json(project_tool_catalog(scenario.get("enabled_tools", {})))


def _format_golden_truth(scenario: dict) -> str:
    """Render the complete golden call sequence as canonical JSON text."""
    return serialize_expected_trace(scenario)


def _observation_metadata(observation: ToolCallObservation) -> dict[str, Any]:
    """Keep execution diagnostics beside, not inside, the evaluated trace."""
    return {
        "observation_status": observation.status,
        "observation_status_reason": observation.status_reason,
        "observation_completed": observation.completed,
        "observation_eos_received": observation.eos_received,
        "observation_transport_error": observation.transport_error,
        "observation_errors": canonical_json(observation.errors),
        "observation_final_response": observation.final_response,
        "observation_thinking": observation.thinking,
        "observation_stream_summary": canonical_json(observation.stream_summary),
        "candidate": canonical_json(observation.candidate),
        "request": canonical_json(observation.request),
    }


def _scenario_metadata(scenario: dict) -> dict:
    """Collect as much scenario metadata as possible for the eval dataset.

    Nested structures (enabled_tools, metadata, golden_truth) and lists
    (expected_tools) are JSON-serialized so the value survives being carried
    through EvalDataset conversation metadata and re-read for reporting.
    The nested ``metadata`` block is also flattened into top-level ``meta_*``
    keys for convenient per-column reporting.
    """
    inner_meta = scenario.get("metadata", {}) or {}
    meta = {
        "scenario_id": scenario.get("scenario_id"),
        "combination_id": scenario.get("combination_id"),
        "variant": scenario.get("variant"),
        # Keep legacy source data inspectable without presenting it as the
        # evaluator's system prompt or sending it to the agent endpoint.
        "source_system_prompt": scenario.get("system_prompt", ""),
        "hop_count": scenario.get("hop_count"),
        "expected_tools": ",".join(scenario.get("expected_tools", [])),
        "enabled_tools": json.dumps(scenario.get("enabled_tools", {}), ensure_ascii=False),
        "golden_truth": json.dumps(scenario.get("golden_truth", []), ensure_ascii=False),
        "tool_catalog": canonical_json(
            project_tool_catalog(scenario.get("enabled_tools", {}))
        ),
        "placeholder_dependencies": canonical_json(
            _placeholder_dependencies(scenario.get("golden_truth", []))
        ),
        "scenario_metadata": json.dumps(inner_meta, ensure_ascii=False),
    }
    # Flatten the nested metadata block into meta_* columns for readability.
    for key, value in inner_meta.items():
        if isinstance(value, (dict, list)):
            meta[f"meta_{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            meta[f"meta_{key}"] = value
    return meta


def build_golden_test_set(scenarios: list[dict]) -> GoldenTestSet:
    """Build a GoldenTestSet keyed by scenario_id, carrying full metadata."""
    golden_pairs = [
        GoldenPair(
            id=scenario["scenario_id"],
            input=scenario["questions"][0].strip(),
            expected_output=_format_golden_truth(scenario),
            metadata=_scenario_metadata(scenario),
        )
        for scenario in scenarios
    ]
    return GoldenTestSet(
        id=datetime.now().strftime("%Y%m%d%H%M%S"),
        golden_pairs=golden_pairs,
    )


def build_trace_eval_dataset(
    scenarios: list[dict],
    observations: Mapping[str, ToolCallObservation],
    *,
    dataset_version_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvalDataset:
    """Build an evaluator dataset from completed canonical tool traces.

    Expected and actual assistant messages contain only canonical trace JSON.
    Scenario, catalog, placeholder, and observation diagnostics are carried in
    conversation metadata. Transport/error/incomplete observations are
    explicitly excluded so an empty actual trace cannot be mistaken for a
    successful abstention; their statuses remain in dataset-level metadata.
    """
    selected_scenarios: list[dict] = []
    responses: dict[str, str] = {}
    excluded: dict[str, dict[str, Any]] = {}

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        observation = observations.get(scenario_id)
        if observation is None:
            excluded[scenario_id] = {
                "status": "missing_observation",
                "reason": "no candidate observation was provided",
            }
            continue
        if observation.status in _EXECUTION_FAILURE_STATUSES or not observation.completed:
            excluded[scenario_id] = {
                "status": observation.status,
                "reason": observation.status_reason,
                "completed": observation.completed,
                "eos_received": observation.eos_received,
            }
            continue

        selected_scenarios.append(scenario)
        responses[scenario_id] = serialize_observed_trace(observation)

    golden_test_set = build_golden_test_set(selected_scenarios)
    dataset_metadata: dict[str, Any] = {
        "dataset_version_id": dataset_version_id,
        "benchmark": "tool_calling",
        "canonical_trace_schema_version": CANONICAL_TRACE_SCHEMA_VERSION,
        "included_scenario_count": len(selected_scenarios),
        "excluded_execution_failures": canonical_json(excluded),
    }
    if metadata:
        dataset_metadata.update(dict(metadata))

    # Add observation-specific metadata to each GoldenPair.  The existing
    # build_eval_dataset boundary copies pair metadata into Conversation
    # metadata and adds expected_output there.
    for scenario in selected_scenarios:
        scenario_id = scenario["scenario_id"]
        observation = observations[scenario_id]
        pair = golden_test_set[scenario_id]
        pair.metadata.update(_observation_metadata(observation))
        pair.metadata["placeholder_dependencies"] = canonical_json(
            _placeholder_dependencies(scenario.get("golden_truth", []))
        )
        # ``tool_call_failure_modes`` formats this evaluator-only metadata as
        # SYSTEM PROMPT. Keep it registry-derived and separate from the
        # agentic request, whose endpoint owns its own instructions/schemas.
        pair.metadata["system_prompt"] = _render_evaluator_tool_catalog(scenario)

    candidate_models = {
        (observation.provider, observation.model)
        for observation in observations.values()
        if observation.scenario_id in responses
    }
    if len(candidate_models) == 1:
        provider, model = next(iter(candidate_models))
        dataset_metadata.setdefault("candidate_provider", provider)
        dataset_metadata.setdefault("candidate_model", model)

    return build_eval_dataset(
        responses=responses,
        dataset=golden_test_set,
        question_func=lambda pair: pair.input,
        metadata=dataset_metadata,
    )


def load_scenarios(dataset_path: Path, limit: int | None = None) -> list[dict]:
    """Load and validate the complete source dataset before applying ``limit``.

    Validation intentionally runs before limiting so an invalid scenario later
    in the canonical file cannot be hidden by a small smoke-run limit.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    validate_scenarios(scenarios)
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios


def build_dry_run_report(
    scenarios: list[dict],
    candidate_models: list[dict],
    *,
    dataset_path: Path | str,
) -> dict[str, Any]:
    """Build a deterministic, credential-free preview of agentic requests.

    Dataset validation is performed by :func:`load_scenarios` before this
    helper is called.  Requests are still built for every selected
    scenario/model pair so dry-run exercises the same request adapter as live
    execution without constructing credentials or opening a socket.
    """
    requests: list[dict[str, Any]] = []
    for model in candidate_models:
        for scenario in scenarios:
            request = build_agentic_request(
                scenario["questions"][0].strip(),
                scenario["enabled_tools"],
                model_name=model["name"],
                model_provider=model["provider"],
            )
            requests.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "model_name": model["name"],
                    "model_provider": model["provider"],
                    "request": redact_sensitive_data(request),
                }
            )

    return {
        "mode": "dry-run",
        "dataset_path": str(dataset_path),
        "scenario_count": len(scenarios),
        "candidate_count": len(candidate_models),
        "request_count": len(requests),
        "requests": requests,
    }


def _print_dry_run_report(report: dict[str, Any]) -> None:
    """Print stable JSON so dry-run output can be inspected or parsed."""
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
