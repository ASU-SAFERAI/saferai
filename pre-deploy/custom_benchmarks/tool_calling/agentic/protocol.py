"""Parser and compact serializer for agentic WebSocket stream frames."""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict, cast

EOS_TOKEN = "<EOS>"


class ToolFunctionPayload(TypedDict, total=False):
    """Function-call fields emitted inside a tool-call event."""

    name: str
    arguments: str


class ToolCallPayload(TypedDict, total=False):
    """Fields used by the agentic stream's heterogeneous tool-call events.

    The stream uses the same envelope for function fragments, progress events,
    and result/render events, so every field is optional. The runtime retains
    additional provider-specific fields in the underlying dictionary.
    """

    type: str
    name: str
    tool: str
    tool_call_id: str
    id: str
    agent_id: str | None
    is_handoff: bool | None
    metadata: dict[str, Any]
    function: ToolFunctionPayload
    content: Any
    result: Any
    render: Any


class ToolCallRecord(TypedDict):
    """Normalized schema stored in ``StreamTrace.tool_calls``."""

    event_type: str | None
    tool_name: str | None
    tool_call_id: str | None
    agent_id: str | None
    is_handoff: bool | None
    metadata: dict[str, Any]
    tool_call_metadata: dict[str, Any]
    payload: ToolCallPayload


@dataclass
class ParsedFrame:
    """A normalized representation of one received WebSocket message."""

    raw: str
    payload: Any
    event_type: str
    text_delta: str | None = None
    thinking: str | None = None
    response: Any = None
    tool_call: ToolCallPayload | None = None
    metadata: dict[str, Any] | None = None
    is_eos: bool = False
    is_error: bool = False
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StreamTrace:
    """Accumulated stream data with compact and optional raw serialization."""

    frames: list[ParsedFrame] = field(default_factory=list)
    text_parts: list[str] = field(default_factory=list)
    thinking_parts: list[str] = field(default_factory=list)
    responses: list[Any] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    eos_metadata: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    eos_received: bool = False
    normal_eos_received: bool = False
    transport_error: str | None = None

    def add(self, raw_message: str | bytes) -> ParsedFrame:
        frame = parse_stream_frame(raw_message)
        self.frames.append(frame)

        if frame.text_delta is not None:
            self.text_parts.append(frame.text_delta)
        if frame.thinking is not None:
            self.thinking_parts.append(frame.thinking)
        if frame.response is not None and not frame.is_eos:
            self.responses.append(frame.response)
        if frame.tool_call is not None:
            self.tool_calls.append(_tool_call_record(frame))
        if frame.error is not None:
            self.errors.append(frame.error)
        if frame.is_eos:
            self.eos_received = True
            if frame.metadata is not None:
                self.eos_metadata = frame.metadata
            self.normal_eos_received = not frame.is_error
        return frame

    def mark_transport_error(self, message: str) -> None:
        self.transport_error = message

    @property
    def completed(self) -> bool:
        """Whether a normal EOS was received without a transport failure."""
        return (
            self.normal_eos_received
            and self.transport_error is None
            and not self.errors
        )

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    @property
    def thinking(self) -> str:
        return "".join(self.thinking_parts)

    @property
    def response(self) -> str:
        """Combine all response fragments into one readable response string."""
        rendered_parts: list[str] = []
        for part in self.responses:
            if isinstance(part, str):
                rendered_parts.append(part)
            else:
                rendered_parts.append(json.dumps(part, ensure_ascii=False))
        return self._merge_stream_fragments(rendered_parts)

    @property
    def event_counts(self) -> dict[str, int]:
        """Count event types without retaining thousands of event envelopes."""
        return dict(Counter(frame.event_type for frame in self.frames))

    @staticmethod
    def _merge_stream_fragments(fragments: list[str]) -> str:
        """Merge delta or cumulative chunks without dropping repeated characters."""
        merged = ""
        for fragment in fragments:
            if not fragment:
                continue
            if not merged:
                merged = fragment
                continue
            if fragment == merged:
                continue
            if fragment.startswith(merged):
                # Cumulative snapshot: the new frame already contains all data.
                merged = fragment
                continue
            if merged.startswith(fragment):
                # Older/shorter cumulative snapshot; keep the longest value.
                continue

            # Merge a genuine boundary overlap, but never treat a full one-
            # character chunk as a duplicate: "a", "a" may legitimately mean
            # "aa" in a delta stream.
            max_overlap = min(len(merged), len(fragment)) - 1
            for overlap in range(max_overlap, 0, -1):
                if merged[-overlap:] == fragment[:overlap]:
                    merged += fragment[overlap:]
                    break
            else:
                merged += fragment
        return merged

    def _find_unidentified_function_group(
        self,
        grouped: dict[str, dict[str, Any]],
    ) -> str | None:
        """Find a prior call whose ID-less function payload is a duplicate."""
        for group_key, batch in reversed(list(grouped.items())):
            if group_key.startswith("event-"):
                continue
            event_types = batch["event_types"]
            if event_types.get("function"):
                return group_key
        return None

    def compact_tool_calls(
        self,
        *,
        include_raw_events: bool = False,
    ) -> list[dict[str, Any]]:
        """Combine streamed tool-call fragments into one record per logical call.

        ``include_raw_events`` is opt-in so the ordinary compact report stays
        small, while normalization/reporting code can retain every provider
        payload that contributed to a logical call.
        """
        grouped: dict[str, dict[str, Any]] = {}
        for event_index, event in enumerate(self.tool_calls):
            call_id = event.get("tool_call_id")
            if call_id:
                group_key = call_id
            elif event.get("event_type") == "function":
                group_key = self._find_unidentified_function_group(grouped)
                if group_key is None:
                    group_key = f"event-{event_index}"
            else:
                group_key = f"event-{event_index}"
            batch = grouped.get(group_key)
            if batch is None:
                batch = {
                    "tool_name": event.get("tool_name"),
                    "tool_call_id": call_id,
                    "agent_id": event.get("agent_id"),
                    "is_handoff": event.get("is_handoff"),
                    "event_count": 0,
                    "event_types": Counter(),
                    "metadata": {},
                    "raw_events": [],
                    "output": {
                        "function_arguments": [],
                        "content": [],
                        "progress": [],
                        "results": [],
                        "renders": [],
                    },
                }
                grouped[group_key] = batch

            batch["event_count"] += 1
            batch["event_types"][event.get("event_type") or "unknown"] += 1
            if include_raw_events:
                batch["raw_events"].append(deepcopy(event))
            metadata = event.get("tool_call_metadata") or event.get("metadata") or {}
            if isinstance(metadata, dict):
                batch["metadata"].update(metadata)

            payload = event.get("payload") or {}
            function_payload = payload.get("function") if isinstance(payload, dict) else None
            if isinstance(function_payload, dict):
                arguments = function_payload.get("arguments")
                if isinstance(arguments, str) and arguments:
                    batch["output"]["function_arguments"].append(arguments)

            content = payload.get("content") if isinstance(payload, dict) else None
            if isinstance(content, str) and content:
                batch["output"]["content"].append(content)
            elif content is not None and content not in batch["output"]["progress"]:
                batch["output"]["progress"].append(content)

            result = payload.get("result") if isinstance(payload, dict) else None
            if result is not None and result not in batch["output"]["results"]:
                batch["output"]["results"].append(result)

            render = payload.get("render") if isinstance(payload, dict) else None
            if render is not None and render not in batch["output"]["renders"]:
                batch["output"]["renders"].append(render)

        compact: list[dict[str, Any]] = []
        for batch in grouped.values():
            batch["event_types"] = dict(batch["event_types"])
            output = batch["output"]
            output["function_arguments"] = self._merge_stream_fragments(
                output["function_arguments"]
            )
            output["content"] = self._merge_stream_fragments(output["content"])
            # Empty collections add noise without providing debugging value.
            batch["output"] = {
                key: value
                for key, value in output.items()
                if value not in ("", [], None)
            }
            if not include_raw_events:
                batch.pop("raw_events", None)
            compact.append(batch)
        return compact

    def to_dict(self, *, include_raw_stream: bool = False) -> dict[str, Any]:
        """Serialize a compact report, optionally including every raw frame."""
        result: dict[str, Any] = {
            "completed": self.completed,
            "eos_received": self.eos_received,
            "normal_eos_received": self.normal_eos_received,
            "text": self.text,
            "thinking": self.thinking,
            "response": self.response,
            "eos_metadata": self.eos_metadata,
            "tool_calls": self.compact_tool_calls(),
            "errors": self.errors,
            "transport_error": self.transport_error,
            "stream_summary": {
                "frame_count": len(self.frames),
                "event_counts": self.event_counts,
            },
        }
        if include_raw_stream:
            result["frames"] = [frame.to_dict() for frame in self.frames]
            result["tool_call_events"] = self.tool_calls
        return result


def _decode_message(raw_message: str | bytes) -> str:
    if isinstance(raw_message, bytes):
        return raw_message.decode("utf-8")
    if isinstance(raw_message, str):
        return raw_message
    raise TypeError(f"Unsupported WebSocket message type: {type(raw_message).__name__}")


def _tool_call_record(frame: ParsedFrame) -> ToolCallRecord:
    tool_call = frame.tool_call or {}
    function_payload = tool_call.get("function")
    tool_metadata: dict[str, Any] = {}
    if isinstance(frame.metadata, dict):
        tool_metadata.update(frame.metadata)
    nested_metadata = tool_call.get("metadata")
    if isinstance(nested_metadata, dict):
        tool_metadata.update(nested_metadata)
    return {
        "event_type": tool_call.get("type"),
        "tool_name": (
            tool_call.get("name")
            or tool_call.get("tool")
            or (function_payload.get("name") if isinstance(function_payload, dict) else None)
        ),
        "tool_call_id": tool_call.get("tool_call_id") or tool_call.get("id"),
        "agent_id": tool_call.get("agent_id"),
        "is_handoff": tool_call.get("is_handoff"),
        "metadata": tool_metadata,
        "tool_call_metadata": tool_metadata,
        "payload": tool_call,
    }


def parse_stream_frame(raw_message: str | bytes) -> ParsedFrame:
    """Parse one protocol frame without discarding unknown fields."""
    raw = _decode_message(raw_message)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ParsedFrame(raw=raw, payload=None, event_type="text", response=raw)

    if not isinstance(payload, dict):
        return ParsedFrame(raw=raw, payload=payload, event_type="json", response=payload)

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = None

    tool_call = payload.get("tool_call")
    if not isinstance(tool_call, dict):
        tool_call = None
    else:
        tool_call = cast(ToolCallPayload, tool_call)

    error_value = payload.get("error")
    error: dict[str, Any] | None = None
    if error_value is not None:
        error = {
            "error": error_value,
            "error_type": payload.get("error_type"),
            "metadata": metadata,
        }

    response = payload.get("response")
    normal_eos = response == EOS_TOKEN
    error_eos = error_value == EOS_TOKEN
    event_type = "eos" if normal_eos else "error_eos" if error_eos else "message"
    if tool_call is not None:
        event_type = tool_call.get("type") or "tool_call"
    elif "text_delta" in payload:
        event_type = "text_delta"
    elif "thinking" in payload:
        event_type = "thinking"
    elif error is not None:
        event_type = "error"

    return ParsedFrame(
        raw=raw,
        payload=payload,
        event_type=event_type,
        text_delta=payload.get("text_delta") if isinstance(payload.get("text_delta"), str) else None,
        thinking=payload.get("thinking") if isinstance(payload.get("thinking"), str) else None,
        response=response,
        tool_call=tool_call,
        metadata=metadata,
        is_eos=normal_eos or error_eos,
        is_error=error is not None,
        error=error,
    )
