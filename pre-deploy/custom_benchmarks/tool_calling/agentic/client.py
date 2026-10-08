"""websocket-client transport for the agentic query stream."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from .config import (
    DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
    WebSocketCredentials,
    redact_sensitive_data,
    redact_sensitive_text,
    validate_websocket_timeout,
)
from .protocol import StreamTrace

# Shared benchmark logger and convention. Guarded so the transport stays usable
# when the module is imported directly (outside the package) for ad-hoc probes.
try:
    from ..benchmark._common import logger
except ImportError:  # pragma: no cover - direct-execution fallback
    import logging

    logger = logging.getLogger("tool_calling_benchmark")

# Emit an INFO-level heartbeat at most this often while a single query streams,
# so a long server-side agent loop is visibly alive without one log line per
# frame. Every frame is still logged at DEBUG.
_HEARTBEAT_INTERVAL_SECONDS = 5.0

try:
    import certifi
except ImportError:  # The caller may provide AGENTIC_CA_BUNDLE instead.
    certifi = None  # type: ignore[assignment]

try:
    import websocket
except ImportError:  # Keep protocol/config utilities usable without the optional transport.
    websocket = None  # type: ignore[assignment]


@dataclass
class WebSocketQueryResult:
    """Request plus its complete parsed stream trace."""

    request: dict[str, Any]
    trace: StreamTrace

    def to_dict(self, *, include_raw_stream: bool = False) -> dict[str, Any]:
        """Serialize the result without exposing request or stream secrets."""
        return redact_sensitive_data({
            "request": self.request,
            "stream": self.trace.to_dict(include_raw_stream=include_raw_stream),
        })


def _redact_exception(exc: Exception) -> str:
    """Render a transport exception without echoing credentials."""
    return redact_sensitive_text(f"{type(exc).__name__}: {exc}")


class AgenticWebSocketClient:
    """Open one authenticated WebSocket per query and parse until EOS."""

    def __init__(
        self,
        credentials: WebSocketCredentials,
        *,
        timeout: float = DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
        sslopt: dict[str, Any] | None = None,
    ) -> None:
        self._credentials = credentials
        self._timeout = validate_websocket_timeout(timeout)
        self._sslopt = dict(sslopt or {})
        if "ca_certs" not in self._sslopt:
            ca_bundle = os.getenv("AGENTIC_CA_BUNDLE")
            if ca_bundle:
                self._sslopt["ca_certs"] = ca_bundle
            elif certifi is not None:
                # Some Conda environments expose a malformed SSL_CERT_FILE to
                # OpenSSL. Explicitly selecting certifi keeps verification on
                # while avoiding that environment-specific default.
                self._sslopt["ca_certs"] = certifi.where()

    def query(self, request: dict[str, Any]) -> WebSocketQueryResult:
        """Send one JSON request and collect frames through normal/error EOS."""
        if not isinstance(request, dict):
            raise TypeError("request must be a dictionary")
        if websocket is None:
            raise RuntimeError(
                "websocket-client is required for live probes; install the pinned project dependency"
            )

        trace = StreamTrace()
        connection = None
        start = time.monotonic()
        frame_count = 0
        last_heartbeat = start
        try:
            connection = websocket.create_connection(
                self._credentials.authenticated_ws_url,
                timeout=self._timeout,
                sslopt=self._sslopt,
            )
            logger.debug("WebSocket connected; sending query and awaiting stream")
            connection.send(json.dumps(request, ensure_ascii=False))
            while not trace.eos_received:
                raw_message = connection.recv()
                if raw_message in (None, "", b""):
                    trace.mark_transport_error("WebSocket closed before EOS")
                    break
                trace.add(raw_message)
                frame_count += 1
                # Per-frame detail at DEBUG; a periodic INFO heartbeat so a long
                # server-side agent loop is visibly alive at the default level.
                logger.debug("Stream frame %d received", frame_count)
                now = time.monotonic()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL_SECONDS:
                    logger.info(
                        "...still streaming: %d frame(s) in %.0fs, awaiting EOS",
                        frame_count,
                        now - start,
                    )
                    last_heartbeat = now
        except Exception as exc:  # Preserve partial frames for engineering inspection.
            trace.mark_transport_error(_redact_exception(exc))
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

        logger.debug(
            "Query finished: %d frame(s) in %.1fs (eos_received=%s)",
            frame_count,
            time.monotonic() - start,
            trace.eos_received,
        )
        return WebSocketQueryResult(request=request, trace=trace)
