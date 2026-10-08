"""Canonical JSON serialization for evaluated tool-call traces.

The benchmark evaluates the ordered ``tool_calls`` value rather than a
candidate's final natural-language response.  This module deliberately keeps
that representation small and deterministic:

* expected traces contain only non-null tool names and arguments;
* observed traces retain parsed arguments, malformed argument diagnostics, and
  each logical call's tool output;
* optional metadata can be carried at the trace level without accidentally
  adding final-response prose to the evaluated value; and
* values outside JSON's native data model are converted deterministically.

Raw provider events and final response/thinking text belong in observation
reports, not in the canonical trace sent to the evaluator.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import math
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Iterable, Mapping, Sequence
from uuid import UUID

from .models import CanonicalCall, GoldenCall, ScenarioView, ToolCallObservation

CANONICAL_TRACE_SCHEMA_VERSION = 1


def _json_safe(value: Any, *, _active: set[int] | None = None) -> Any:
    """Convert a nested value to JSON-compatible data deterministically.

    Tool arguments and tool results are normally JSON-native, but metadata and
    provider payloads can contain values such as ``Decimal``, ``bytes``, sets,
    paths, or datetimes.  Unsupported values use a type-tagged string form so
    serialization remains loss-aware and never depends on an object's memory
    address.  Cyclic containers are rejected with a useful error instead of
    recursing forever.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__type__": "float", "value": str(value)}
    if isinstance(value, Decimal):
        return {"__type__": "decimal", "value": str(value)}
    if isinstance(value, (datetime, date, time)):
        return {"__type__": type(value).__name__, "value": value.isoformat()}
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, Enum):
        return _json_safe(value.value, _active=_active)

    active = _active if _active is not None else set()
    is_container = isinstance(value, (Mapping, list, tuple, set, frozenset)) or dataclasses.is_dataclass(value)
    if is_container:
        object_id = id(value)
        if object_id in active:
            raise TypeError("cannot canonicalize a cyclic value")
        active.add(object_id)
        try:
            if dataclasses.is_dataclass(value) and not isinstance(value, type):
                value = dataclasses.asdict(value)

            if isinstance(value, Mapping):
                # JSON object keys are strings.  Sorting is delegated to the
                # final json.dumps call, while this conversion makes unusual
                # key types safe and deterministic first.
                return {
                    str(key): _json_safe(item, _active=active)
                    for key, item in value.items()
                }
            if isinstance(value, (list, tuple)):
                return [_json_safe(item, _active=active) for item in value]
            if isinstance(value, (set, frozenset)):
                converted = [_json_safe(item, _active=active) for item in value]
                return sorted(converted, key=_sort_key)
        finally:
            active.remove(object_id)

    # ``repr`` is intentionally not used: it commonly includes memory
    # addresses.  The class-qualified string form is stable for ordinary
    # provider objects and still makes the fallback visible to reviewers.
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    return {"__type__": type_name, "value": str(value)}


def _sort_key(value: Any) -> str:
    """Return a stable key for sorting converted set members."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any) -> str:
    """Render JSON with stable key ordering and compact, UTF-8-safe formatting."""
    safe_value = _json_safe(value)
    return json.dumps(
        safe_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _expected_source(source: ScenarioView | Mapping[str, Any] | Sequence[Any]) -> Iterable[Any]:
    if isinstance(source, ScenarioView):
        return source.golden_truth
    if isinstance(source, Mapping):
        if "golden_truth" in source:
            value = source["golden_truth"]
        else:
            value = source.get("tool_calls", [])
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError("expected trace golden_truth must be a sequence")
        return value
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError("expected trace must be a ScenarioView, mapping, or sequence")
    return source


def _call_value(call: Any, key: str, default: Any = None) -> Any:
    if isinstance(call, GoldenCall) or isinstance(call, CanonicalCall):
        return getattr(call, key, default)
    if isinstance(call, Mapping):
        return call.get(key, default)
    raise TypeError("trace calls must be GoldenCall, CanonicalCall, or mapping objects")


def expected_trace_payload(
    source: ScenarioView | Mapping[str, Any] | Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical payload for an expected golden trace.

    Explicit null-tool records are the source representation for abstention;
    they become an empty ``tool_calls`` list in the evaluated representation.
    Symbolic strings such as ``{{steps[0].output}}`` are copied unchanged.
    """
    calls: list[dict[str, Any]] = []
    for call in _expected_source(source):
        tool_name = _call_value(call, "tool_name")
        if tool_name is None:
            continue
        calls.append(
            {
                "tool_name": _json_safe(tool_name),
                "arguments": _json_safe(_call_value(call, "arguments", {})),
            }
        )

    payload: dict[str, Any] = {"tool_calls": calls}
    if metadata is not None:
        payload["metadata"] = _json_safe(metadata)
    return payload


def serialize_expected_trace(
    source: ScenarioView | Mapping[str, Any] | Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Serialize an expected golden trace as canonical JSON text."""
    return canonical_json(expected_trace_payload(source, metadata=metadata))


def _observed_source(
    source: ToolCallObservation | Mapping[str, Any] | Sequence[Any],
) -> tuple[Iterable[Any], Mapping[str, Any] | None]:
    if isinstance(source, ToolCallObservation):
        return source.tool_calls, None
    if isinstance(source, Mapping):
        calls = source.get("tool_calls", [])
        if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
            raise TypeError("observed trace tool_calls must be a sequence")
        return calls, None
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError("observed trace must be a ToolCallObservation, mapping, or sequence")
    return source, None


def _observed_call_payload(call: Any, ordinal: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ordinal": _json_safe(_call_value(call, "ordinal", ordinal)),
        "tool_name": _json_safe(_call_value(call, "tool_name")),
        "arguments": _json_safe(_call_value(call, "arguments")),
        "tool_output": _json_safe(_call_value(call, "tool_output")),
    }

    # Invalid JSON arguments remain visible without pretending that their raw
    # text is a parsed object.  These fields are omitted for ordinary calls to
    # keep equivalent traces compact and stable.
    argument_text = _call_value(call, "argument_text")
    parse_error = _call_value(call, "argument_parse_error")
    tool_call_id = _call_value(call, "tool_call_id")
    event_type = _call_value(call, "event_type")
    call_metadata = _call_value(call, "metadata", {})
    if argument_text is not None:
        payload["argument_text"] = _json_safe(argument_text)
    if parse_error is not None:
        payload["argument_parse_error"] = _json_safe(parse_error)
    if tool_call_id is not None:
        payload["tool_call_id"] = _json_safe(tool_call_id)
    if event_type is not None:
        payload["event_type"] = _json_safe(event_type)
    if call_metadata:
        payload["metadata"] = _json_safe(call_metadata)
    return payload


def observed_trace_payload(
    source: ToolCallObservation | Mapping[str, Any] | Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical payload for an observed ordered tool-call trace.

    The observation's final response and thinking text are intentionally not
    copied.  They are report-only fields and must not replace the evaluated
    tool-call trace.  Use ``metadata`` for evaluator-safe context that belongs
    beside the trace rather than inside a call.
    """
    calls, _ = _observed_source(source)
    payload: dict[str, Any] = {
        "tool_calls": [
            _observed_call_payload(call, ordinal)
            for ordinal, call in enumerate(calls)
        ]
    }
    if metadata is not None:
        payload["metadata"] = _json_safe(metadata)
    return payload


def serialize_observed_trace(
    source: ToolCallObservation | Mapping[str, Any] | Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Serialize an observed trace as canonical JSON text."""
    return canonical_json(observed_trace_payload(source, metadata=metadata))


def audit_observed_trace_payload(
    source: ToolCallObservation | Mapping[str, Any] | Sequence[Any],
) -> dict[str, Any]:
    """Build a human-readable observed trace with only tool name and arguments.

    The full :func:`observed_trace_payload` retains diagnostics the evaluator
    needs (``ordinal``, ``tool_output``, raw ``argument_text``, provider
    ``metadata``, and ``tool_call_id``).  Those fields make the on-disk audit
    column unreadable: a single web-search call can drag in kilobytes of
    scraped page content.  This projection keeps each call to ``tool_name`` and
    parsed ``arguments`` so a reviewer sees what the model chose to call and
    with which arguments, mirroring the shape of the expected/golden trace.

    A malformed call whose arguments could not be parsed keeps its raw
    ``argument_text`` (and the parse error) so the failure stays visible instead
    of being silently dropped.
    """
    calls, _ = _observed_source(source)
    projected: list[dict[str, Any]] = []
    for call in calls:
        entry: dict[str, Any] = {
            "tool_name": _json_safe(_call_value(call, "tool_name")),
            "arguments": _json_safe(_call_value(call, "arguments")),
        }
        parse_error = _call_value(call, "argument_parse_error")
        if parse_error is not None:
            entry["argument_parse_error"] = _json_safe(parse_error)
            argument_text = _call_value(call, "argument_text")
            if argument_text is not None:
                entry["argument_text"] = _json_safe(argument_text)
        projected.append(entry)
    return {"tool_calls": projected}


def serialize_audit_observed_trace(
    source: ToolCallObservation | Mapping[str, Any] | Sequence[Any],
) -> str:
    """Serialize the pared-down, human-readable observed trace as canonical JSON."""
    return canonical_json(audit_observed_trace_payload(source))


# Descriptive aliases for callers that prefer noun-first names.
canonical_expected_trace = serialize_expected_trace
canonical_observed_trace = serialize_observed_trace


__all__ = [
    "CANONICAL_TRACE_SCHEMA_VERSION",
    "audit_observed_trace_payload",
    "canonical_expected_trace",
    "canonical_json",
    "canonical_observed_trace",
    "expected_trace_payload",
    "observed_trace_payload",
    "serialize_audit_observed_trace",
    "serialize_expected_trace",
    "serialize_observed_trace",
]
