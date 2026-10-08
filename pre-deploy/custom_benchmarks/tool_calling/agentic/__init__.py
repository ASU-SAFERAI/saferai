"""WebSocket probes for the agentic tool-calling API."""

from .client import AgenticWebSocketClient, WebSocketQueryResult
from .config import (
    DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
    MAX_WEBSOCKET_TIMEOUT_SECONDS,
    WebSocketCredentials,
    build_authenticated_ws_url,
    load_credentials,
    redact_sensitive_data,
    redact_sensitive_text,
    validate_websocket_timeout,
)
from .protocol import EOS_TOKEN, StreamTrace, parse_stream_frame
from .observation import (
    TraceStatusClassification,
    build_tool_call_observation,
    canonical_calls_from_compact,
    classify_trace_status,
    extract_canonical_calls,
    extract_tool_call_observations,
)

__all__ = [
    "AgenticWebSocketClient",
    "EOS_TOKEN",
    "StreamTrace",
    "WebSocketCredentials",
    "WebSocketQueryResult",
    "DEFAULT_WEBSOCKET_TIMEOUT_SECONDS",
    "MAX_WEBSOCKET_TIMEOUT_SECONDS",
    "build_authenticated_ws_url",
    "load_credentials",
    "redact_sensitive_data",
    "redact_sensitive_text",
    "validate_websocket_timeout",
    "parse_stream_frame",
    "TraceStatusClassification",
    "build_tool_call_observation",
    "canonical_calls_from_compact",
    "classify_trace_status",
    "extract_canonical_calls",
    "extract_tool_call_observations",
]
