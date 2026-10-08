"""Typed internal models for agentic tool-calling benchmark data.

These models provide a small, source-compatible boundary between the populated
JSON seed, the agentic stream normalizer, and later reporting/scoring code.
They validate the shape of one scenario/call/observation, while dataset-wide
registry and dependency rules remain in ``dataset_validation``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeAlias


TraceStatus: TypeAlias = Literal[
    "completed",
    "abstention",
    "completed_no_call",
    "malformed_arguments",
    "error_eos",
    "transport_error",
    "incomplete",
]

TRACE_STATUSES = frozenset(
    {
        "completed",
        "abstention",
        "completed_no_call",
        "malformed_arguments",
        "error_eos",
        "transport_error",
        "incomplete",
    }
)


def _copy_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    """Copy a mapping so model instances do not alias source JSON objects."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return deepcopy(dict(value))


def _copy_mapping_list(value: Sequence[Mapping[str, Any]], field_name: str) -> list[dict[str, Any]]:
    """Copy a sequence of mapping records used by error and event fields."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field_name} must be a list of objects")
    copied: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{field_name}[{index}] must be an object")
        copied.append(deepcopy(dict(item)))
    return copied


def _require_string(value: Any, field_name: str, *, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


@dataclass
class GoldenCall:
    """One expected tool call, including explicit null-tool abstentions."""

    tool_name: str | None
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_string(self.tool_name, "tool_name", allow_none=True)
        self.arguments = _copy_mapping(self.arguments, "arguments")
        self.raw = _copy_mapping(self.raw, "raw")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoldenCall":
        """Create a call without discarding unknown source fields."""
        source = _copy_mapping(value, "golden call")
        return cls(
            tool_name=source.get("tool_name"),
            arguments=source.get("arguments", {}),
            raw=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))


@dataclass
class ScenarioView:
    """Validated, typed view of one populated benchmark scenario.

    ``raw_record`` retains legacy fields such as ``system_prompt``.  The
    benchmark can therefore inspect or report source data without making that
    legacy prompt part of a future agentic request.
    """

    scenario_id: str
    question: str
    enabled_tools: list[str]
    golden_truth: list[GoldenCall]
    expected_tools: list[str]
    hop_count: int
    combination_id: str | None = None
    variant: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_catalog: dict[str, Any] = field(default_factory=dict)
    raw_record: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_string(self.scenario_id, "scenario_id")
        _require_string(self.question, "question")
        _require_string(self.combination_id, "combination_id", allow_none=True)
        if self.variant is not None and (isinstance(self.variant, bool) or not isinstance(self.variant, int)):
            raise TypeError("variant must be an integer or None")
        if isinstance(self.hop_count, bool) or not isinstance(self.hop_count, int):
            raise TypeError("hop_count must be an integer")
        if self.hop_count < 0:
            raise ValueError("hop_count must be non-negative")

        if not isinstance(self.enabled_tools, list) or not all(
            isinstance(tool_name, str) for tool_name in self.enabled_tools
        ):
            raise TypeError("enabled_tools must be a list of strings")
        if not isinstance(self.expected_tools, list) or not all(
            isinstance(tool_name, str) for tool_name in self.expected_tools
        ):
            raise TypeError("expected_tools must be a list of strings")
        if not isinstance(self.golden_truth, list) or not all(
            isinstance(call, GoldenCall) for call in self.golden_truth
        ):
            raise TypeError("golden_truth must be a list of GoldenCall objects")

        self.enabled_tools = list(self.enabled_tools)
        self.expected_tools = list(self.expected_tools)
        self.golden_truth = list(self.golden_truth)
        self.metadata = _copy_mapping(self.metadata, "metadata")
        self.tool_catalog = _copy_mapping(self.tool_catalog, "tool_catalog")
        self.raw_record = _copy_mapping(self.raw_record, "raw_record")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        tool_order: Sequence[str] | None = None,
    ) -> "ScenarioView":
        """Create a view from either the migrated seed or a compatible record.

        The current seed stores its user text under ``questions`` and enabled
        tools as a boolean mapping.  A singular ``question`` and an already
        projected enabled-tool list are also accepted for compatibility.
        """
        source = _copy_mapping(value, "scenario")

        question = source.get("question")
        if question is None:
            questions = source.get("questions")
            if isinstance(questions, Sequence) and not isinstance(questions, (str, bytes)) and questions:
                question = questions[0]
        if not isinstance(question, str):
            raise TypeError("scenario question must be a string")

        enabled_source = source.get("enabled_tools", {})
        if isinstance(enabled_source, Mapping):
            enabled_names = [
                tool_name
                for tool_name, enabled in enabled_source.items()
                if enabled is True
            ]
            if tool_order is not None:
                ordered = [tool_name for tool_name in tool_order if enabled_source.get(tool_name) is True]
                ordered.extend(
                    tool_name
                    for tool_name in enabled_names
                    if tool_name not in ordered
                )
                enabled_names = ordered
        elif isinstance(enabled_source, list):
            enabled_names = list(enabled_source)
        else:
            raise TypeError("enabled_tools must be an object or list")

        golden_source = source.get("golden_truth", [])
        if not isinstance(golden_source, list):
            raise TypeError("golden_truth must be a list")
        golden_truth = [
            call if isinstance(call, GoldenCall) else GoldenCall.from_mapping(call)
            for call in golden_source
        ]

        expected_tools = source.get("expected_tools", [])
        if not isinstance(expected_tools, list):
            raise TypeError("expected_tools must be a list")

        return cls(
            scenario_id=source.get("scenario_id"),
            question=question,
            enabled_tools=enabled_names,
            golden_truth=golden_truth,
            expected_tools=list(expected_tools),
            hop_count=source.get("hop_count"),
            combination_id=source.get("combination_id"),
            variant=source.get("variant"),
            metadata=source.get("metadata", {}),
            tool_catalog=source.get("tool_catalog", {}),
            raw_record=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))


@dataclass
class CanonicalCall:
    """One ordered observed call after stream fragments are grouped."""

    ordinal: int
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    argument_text: str | None = None
    argument_parse_error: str | None = None
    tool_call_id: str | None = None
    event_type: str | None = None
    tool_output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 0:
            raise ValueError("ordinal must be a non-negative integer")
        _require_string(self.tool_name, "tool_name", allow_none=True)
        _require_string(self.argument_text, "argument_text", allow_none=True)
        _require_string(self.argument_parse_error, "argument_parse_error", allow_none=True)
        _require_string(self.tool_call_id, "tool_call_id", allow_none=True)
        _require_string(self.event_type, "event_type", allow_none=True)
        if self.arguments is not None:
            self.arguments = _copy_mapping(self.arguments, "arguments")
        self.metadata = _copy_mapping(self.metadata, "metadata")
        self.tool_output = deepcopy(self.tool_output)
        self.raw = deepcopy(self.raw)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, ordinal: int | None = None) -> "CanonicalCall":
        """Create a call while retaining the complete compact/event payload."""
        source = _copy_mapping(value, "canonical call")
        call_ordinal = source.get("ordinal", ordinal)
        if call_ordinal is None:
            raise TypeError("canonical call ordinal is required")
        return cls(
            ordinal=call_ordinal,
            tool_name=source.get("tool_name"),
            arguments=source.get("arguments"),
            argument_text=source.get("argument_text"),
            argument_parse_error=source.get("argument_parse_error"),
            tool_call_id=source.get("tool_call_id"),
            event_type=source.get("event_type"),
            tool_output=source.get("tool_output"),
            metadata=source.get("metadata", {}),
            raw=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))


@dataclass
class ToolCallObservation:
    """Serializable result of one candidate/model scenario execution."""

    run_id: str
    scenario_id: str
    candidate: dict[str, str]
    request: dict[str, Any]
    status: TraceStatus
    completed: bool
    eos_received: bool
    tool_calls: list[CanonicalCall] = field(default_factory=list)
    status_reason: str = ""
    final_response: str = ""
    thinking: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)
    transport_error: str | None = None
    stream_summary: dict[str, Any] = field(default_factory=dict)
    raw_trace: dict[str, Any] | None = None
    eos_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        _require_string(self.run_id, "run_id")
        _require_string(self.scenario_id, "scenario_id")
        if self.status not in TRACE_STATUSES:
            allowed = ", ".join(sorted(TRACE_STATUSES))
            raise ValueError(f"status must be one of: {allowed}")
        if not isinstance(self.completed, bool):
            raise TypeError("completed must be a boolean")
        if not isinstance(self.eos_received, bool):
            raise TypeError("eos_received must be a boolean")
        _require_string(self.status_reason, "status_reason")
        _require_string(self.final_response, "final_response")
        _require_string(self.thinking, "thinking")
        _require_string(self.transport_error, "transport_error", allow_none=True)
        if self.eos_metadata is not None:
            self.eos_metadata = _copy_mapping(self.eos_metadata, "eos_metadata")

        self.candidate = _copy_mapping(self.candidate, "candidate")
        for key in ("name", "provider"):
            _require_string(self.candidate.get(key), f"candidate[{key!r}]")
        self.request = _copy_mapping(self.request, "request")
        if not isinstance(self.tool_calls, list) or not all(
            isinstance(call, CanonicalCall) for call in self.tool_calls
        ):
            raise TypeError("tool_calls must be a list of CanonicalCall objects")
        self.tool_calls = list(self.tool_calls)
        self.errors = _copy_mapping_list(self.errors, "errors")
        self.stream_summary = _copy_mapping(self.stream_summary, "stream_summary")
        if self.raw_trace is not None:
            self.raw_trace = _copy_mapping(self.raw_trace, "raw_trace")

    @property
    def model(self) -> str:
        """Candidate model name, convenient for report construction."""
        return self.candidate["name"]

    @property
    def provider(self) -> str:
        """Candidate provider name, convenient for report construction."""
        return self.candidate["provider"]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ToolCallObservation":
        """Create an observation from a serialized observation dictionary."""
        source = _copy_mapping(value, "observation")
        candidate = source.get("candidate")
        if candidate is None:
            candidate = {
                "name": source.get("model"),
                "provider": source.get("provider"),
            }
        tool_calls_source = source.get("tool_calls", [])
        if not isinstance(tool_calls_source, list):
            raise TypeError("tool_calls must be a list")
        tool_calls = [
            call if isinstance(call, CanonicalCall) else CanonicalCall.from_mapping(call, ordinal=index)
            for index, call in enumerate(tool_calls_source)
        ]
        return cls(
            run_id=source.get("run_id"),
            scenario_id=source.get("scenario_id"),
            candidate=candidate,
            request=source.get("request", {}),
            status=source.get("status"),
            completed=source.get("completed"),
            eos_received=source.get("eos_received"),
            eos_metadata=source.get("eos_metadata"),
            tool_calls=tool_calls,
            status_reason=source.get("status_reason", ""),
            final_response=source.get("final_response", ""),
            thinking=source.get("thinking", ""),
            errors=source.get("errors", []),
            transport_error=source.get("transport_error"),
            stream_summary=source.get("stream_summary", {}),
            raw_trace=source.get("raw_trace"),
        )

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(asdict(self))


__all__ = [
    "CanonicalCall",
    "GoldenCall",
    "ScenarioView",
    "TRACE_STATUSES",
    "TraceStatus",
    "ToolCallObservation",
]
