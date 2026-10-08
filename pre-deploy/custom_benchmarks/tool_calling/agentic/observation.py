"""Convert compact agentic stream calls into canonical benchmark calls."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from custom_benchmarks.tool_calling.models import (
    CanonicalCall,
    ToolCallObservation,
    TraceStatus,
)

from .client import WebSocketQueryResult
from .config import redact_sensitive_data
from .protocol import StreamTrace


def _merged_argument_text(value: Any) -> str | None:
    """Return compact function arguments as one text fragment.

    ``StreamTrace.compact_tool_calls`` normally returns a merged string.  The
    list handling keeps this boundary tolerant of hand-built compact records
    and older protocol adapters that expose the fragments directly.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(fragment, str) for fragment in value):
        return "".join(value)
    return None


def _parse_arguments(argument_text: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse one merged function-argument string, retaining useful errors."""
    if argument_text is None:
        return None, None
    try:
        parsed = json.loads(argument_text)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return None, "TypeError: function arguments must decode to a JSON object"
    return parsed, None


def _tool_output(output: Mapping[str, Any]) -> Any:
    """Select the logical result while retaining non-result stream output."""
    results = output.get("results")
    if isinstance(results, list) and results:
        if len(results) == 1:
            return deepcopy(results[0])
        return deepcopy(results)

    diagnostics = {
        key: deepcopy(output[key])
        for key in ("content", "progress", "renders")
        if key in output and output[key] not in (None, "", [])
    }
    return diagnostics or None


def canonical_calls_from_compact(
    compact_calls: Iterable[Mapping[str, Any]],
    *,
    include_raw_payloads: bool = True,
) -> list[CanonicalCall]:
    """Build ordered :class:`CanonicalCall` values from compact stream calls.

    Compact records have already grouped argument fragments by logical call
    and retain their source order.  This adapter deliberately does not dedupe
    by tool name or ID: repeated calls remain separate observations.  Invalid
    JSON is represented by ``arguments=None`` plus the original argument text
    and a parse error, while result/progress/render output and provider event
    payloads remain available for forensic reporting.
    """
    canonical: list[CanonicalCall] = []
    for ordinal, compact in enumerate(compact_calls):
        if not isinstance(compact, Mapping):
            raise TypeError(f"compact_calls[{ordinal}] must be an object")

        output = compact.get("output", {})
        if not isinstance(output, Mapping):
            output = {}
        argument_text = _merged_argument_text(output.get("function_arguments"))
        arguments, parse_error = _parse_arguments(argument_text)

        event_types = compact.get("event_types", {})
        if isinstance(event_types, Mapping):
            event_type_names = [str(name) for name in event_types]
        else:
            event_type_names = []
        event_type = (
            event_type_names[0]
            if len(event_type_names) == 1
            else "mixed"
            if event_type_names
            else None
        )

        metadata = compact.get("metadata", {})
        metadata = deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
        for key in ("event_count", "event_types", "agent_id", "is_handoff"):
            if key in compact:
                metadata[key] = deepcopy(compact[key])
        if output:
            metadata["stream_output"] = deepcopy(dict(output))

        canonical.append(
            CanonicalCall(
                ordinal=ordinal,
                tool_name=compact.get("tool_name"),
                arguments=arguments,
                argument_text=argument_text,
                argument_parse_error=parse_error,
                tool_call_id=compact.get("tool_call_id"),
                event_type=event_type,
                tool_output=_tool_output(output),
                metadata=metadata,
                raw=deepcopy(compact) if include_raw_payloads else None,
            )
        )
    return canonical


def extract_canonical_calls(
    stream: StreamTrace,
    *,
    include_raw_payloads: bool = True,
) -> list[CanonicalCall]:
    """Extract ordered canonical calls from a collected ``StreamTrace``."""
    return canonical_calls_from_compact(
        stream.compact_tool_calls(include_raw_events=include_raw_payloads),
        include_raw_payloads=include_raw_payloads,
    )




@dataclass(frozen=True)
class TraceStatusClassification:
    """Deterministic status and completion result for one collected trace."""

    status: TraceStatus
    completed: bool
    status_reason: str


_COMPLETED_STATUSES = frozenset({"completed", "abstention", "completed_no_call"})


def classify_trace_status(
    stream: StreamTrace,
    *,
    expected_call_count: int | None = None,
    calls: Sequence[CanonicalCall] | None = None,
) -> TraceStatusClassification:
    """Classify a stream using the benchmark's fixed status precedence.

    Precedence is transport failure, stream/error EOS, malformed arguments,
    normal-EOS call shape, and finally incomplete/no-EOS.  ``expected_call_count``
    is zero only for an abstention scenario; omitted/positive values make a
    completed empty trace ``completed_no_call``.  The optional ``calls`` value
    lets callers reuse extraction work from :func:`extract_canonical_calls`.
    """
    if expected_call_count is not None and (
        isinstance(expected_call_count, bool)
        or not isinstance(expected_call_count, int)
        or expected_call_count < 0
    ):
        raise ValueError("expected_call_count must be a non-negative integer or None")

    observed_calls = list(calls) if calls is not None else extract_canonical_calls(
        stream,
        include_raw_payloads=False,
    )

    # Transport failures take precedence even when an error frame or partial
    # malformed call was received before the connection closed.
    if stream.transport_error:
        status: TraceStatus = "transport_error"
        return TraceStatusClassification(
            status=status,
            completed=False,
            status_reason="transport error prevented normal completion",
        )

    # ``errors`` includes provider stream errors.  An error EOS is also
    # recognized from the EOS flags so hand-built traces cannot be mistaken for
    # an incomplete trace merely because the error payload was not retained.
    if stream.errors or (stream.eos_received and not stream.normal_eos_received):
        status = "error_eos"
        return TraceStatusClassification(
            status=status,
            completed=False,
            status_reason="stream error or error EOS prevented normal completion",
        )

    if any(call.argument_parse_error for call in observed_calls):
        status = "malformed_arguments"
        return TraceStatusClassification(
            status=status,
            completed=False,
            status_reason="one or more tool calls contain malformed JSON arguments",
        )

    if stream.normal_eos_received:
        if observed_calls:
            status = "completed"
            reason = "normal EOS received with one or more valid tool calls"
        elif expected_call_count == 0:
            status = "abstention"
            reason = "normal EOS received with no tool calls for an abstention scenario"
        else:
            status = "completed_no_call"
            reason = "normal EOS received with no tool calls for a scenario requiring a tool call"
        return TraceStatusClassification(
            status=status,
            completed=status in _COMPLETED_STATUSES,
            status_reason=reason,
        )

    return TraceStatusClassification(
        status="incomplete",
        completed=False,
        status_reason="stream ended before normal EOS",
    )


def build_tool_call_observation(
    result: WebSocketQueryResult,
    *,
    run_id: str,
    scenario_id: str,
    candidate: Mapping[str, str],
    expected_call_count: int | None = None,
    include_raw_frames: bool = False,
) -> ToolCallObservation:
    """Build a complete, redacted observation from one WebSocket result.

    Extraction is performed once with the requested raw-event setting and the
    resulting calls are passed into the fixed status classifier.  Request,
    error, EOS, and forensic stream payloads are redacted before they cross
    the observation boundary.  Raw frames and compact call payloads are
    retained only when ``include_raw_frames`` is true; parsed arguments and
    tool outputs remain available for evaluation in either mode.
    """
    if not isinstance(result, WebSocketQueryResult):
        raise TypeError("result must be a WebSocketQueryResult")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a non-empty string")
    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be an object with name and provider")
    if not isinstance(include_raw_frames, bool):
        raise TypeError("include_raw_frames must be a boolean")

    stream = result.trace
    calls = extract_canonical_calls(
        stream,
        include_raw_payloads=include_raw_frames,
    )
    classification = classify_trace_status(
        stream,
        expected_call_count=expected_call_count,
        calls=calls,
    )

    # ``CanonicalCall.raw`` and metadata are forensic provider data.  Redact
    # these fields in place while leaving parsed arguments and tool outputs
    # intact for the evaluator's placeholder/dependency checks.
    for call in calls:
        if call.metadata:
            call.metadata = cast(dict[str, Any], redact_sensitive_data(call.metadata))
        if call.raw is not None:
            call.raw = redact_sensitive_data(call.raw)

    raw_trace: dict[str, Any] | None = None
    if include_raw_frames:
        serialized_trace = redact_sensitive_data(
            stream.to_dict(include_raw_stream=True)
        )
        raw_trace = cast(dict[str, Any], serialized_trace)

    redacted_request = redact_sensitive_data(result.request)
    if not isinstance(redacted_request, dict):  # pragma: no cover - typed input guard
        raise TypeError("result request must be an object")
    redacted_errors = redact_sensitive_data(stream.errors)
    if not isinstance(redacted_errors, list):  # pragma: no cover - typed input guard
        raise TypeError("stream errors must be a list")
    redacted_eos_metadata = (
        redact_sensitive_data(stream.eos_metadata)
        if stream.eos_metadata is not None
        else None
    )
    if redacted_eos_metadata is not None and not isinstance(redacted_eos_metadata, dict):
        raise TypeError("stream EOS metadata must be an object")

    return ToolCallObservation(
        run_id=run_id,
        scenario_id=scenario_id,
        candidate=dict(candidate),
        request=cast(dict[str, Any], redacted_request),
        status=classification.status,
        completed=classification.completed,
        eos_received=stream.eos_received,
        eos_metadata=cast(dict[str, Any] | None, redacted_eos_metadata),
        tool_calls=calls,
        status_reason=classification.status_reason,
        final_response=stream.response or stream.text,
        thinking=stream.thinking,
        errors=cast(list[dict[str, Any]], redacted_errors),
        transport_error=(
            cast(str, redact_sensitive_data(stream.transport_error))
            if stream.transport_error is not None
            else None
        ),
        stream_summary={
            "frame_count": len(stream.frames),
            "event_counts": stream.event_counts,
            "tool_call_count": len(stream.tool_calls),
            "eos_received": stream.eos_received,
            "normal_eos_received": stream.normal_eos_received,
            "completed": stream.completed,
        },
        raw_trace=raw_trace,
    )


# Noun-first alias for callers that use the design document terminology.
extract_tool_call_observations = extract_canonical_calls


__all__ = [
    "TraceStatusClassification",
    "build_tool_call_observation",
    "canonical_calls_from_compact",
    "classify_trace_status",
    "extract_canonical_calls",
    "extract_tool_call_observations",
]
