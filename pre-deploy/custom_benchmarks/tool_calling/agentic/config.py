"""Configuration and safe WebSocket URL construction for agentic probes."""

from __future__ import annotations

import configparser
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

DEFAULT_CONFIG_PATH = Path(__file__).with_name("varun_credentials.conf")
DEFAULT_ENVIRONMENT = "DEV"
DEFAULT_WEBSOCKET_TIMEOUT_SECONDS = 120.0
MAX_WEBSOCKET_TIMEOUT_SECONDS = 600.0
REDACTED_VALUE = "<redacted>"

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:access[_-]?token|api[_-]?key|authorization|bearer|credential|password|secret|token)",
    re.IGNORECASE,
)
_SECRET_QUERY_PATTERN = re.compile(
    r"(?i)((?:[?&])?(?:access[_-]?token|api[_-]?key|token|secret)\s*[:=]\s*)[^\s&,]+"
)
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")


def redact_sensitive_text(value: str) -> str:
    """Redact tokens commonly present in URLs, headers, and exceptions."""
    if not isinstance(value, str):
        return value
    redacted = _SECRET_QUERY_PATTERN.sub(rf"\1{REDACTED_VALUE}", value)
    return _BEARER_PATTERN.sub(rf"\1{REDACTED_VALUE}", redacted)


def redact_sensitive_data(value: Any) -> Any:
    """Return a JSON-compatible copy with secret-bearing values redacted."""
    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE
            if _SENSITIVE_KEY_PATTERN.search(str(key))
            else redact_sensitive_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def validate_websocket_timeout(timeout: float) -> float:
    """Validate and return a finite, positive, bounded socket timeout."""
    if isinstance(timeout, bool):
        raise ValueError("WebSocket timeout must be a finite number")
    try:
        normalized = float(timeout)
    except (TypeError, ValueError) as exc:
        raise ValueError("WebSocket timeout must be a finite number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("WebSocket timeout must be greater than zero")
    if normalized > MAX_WEBSOCKET_TIMEOUT_SECONDS:
        raise ValueError(
            f"WebSocket timeout must not exceed {MAX_WEBSOCKET_TIMEOUT_SECONDS:g} seconds"
        )
    return normalized


@dataclass(frozen=True)
class WebSocketCredentials:
    """Credentials needed to open an authenticated WebSocket connection.

    The token is intentionally kept in memory only and is never included in
    the serializable probe report or in the object's representation.
    """

    access_token: str
    authenticated_ws_url: str

    def __repr__(self) -> str:  # pragma: no cover - defensive secret handling
        return "WebSocketCredentials(access_token=<redacted>, authenticated_ws_url=<redacted>)"


def build_authenticated_ws_url(ws_url: str, access_token: str) -> str:
    """Insert ``access_token`` into a WebSocket URL exactly once.

    The checked-in configuration uses a URL ending in ``?access_token=``.
    This function also accepts a bare URL or a URL with an existing query
    parameter, replacing only its access-token value.
    """
    if not ws_url or not ws_url.strip():
        raise ValueError("ws_url must not be empty")
    if not access_token or not access_token.strip():
        raise ValueError("access_token must not be empty")

    parts = urlsplit(ws_url.strip())
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError("ws_url must be an absolute ws:// or wss:// URL")

    query = parse_qsl(parts.query, keep_blank_values=True)
    replaced = False
    authenticated_query: list[tuple[str, str]] = []
    for key, value in query:
        if key == "access_token":
            if not replaced:
                authenticated_query.append((key, access_token))
                replaced = True
            continue
        authenticated_query.append((key, value))

    if not replaced:
        authenticated_query.append(("access_token", access_token))

    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(authenticated_query), parts.fragment)
    )


def _read_config_value(
    parser: configparser.ConfigParser,
    section: str,
    key: str,
    *,
    override: str | None,
) -> str:
    if override is not None:
        value = override.strip()
    elif parser.has_option(section, key):
        value = parser.get(section, key).strip()
    else:
        value = ""
    if not value:
        raise ValueError(f"Missing {key!r} in [{section}] or its environment override")
    return value


def load_credentials(
    config_path: str | Path | None = None,
    environment: str | None = None,
    *,
    access_token: str | None = None,
    ws_url: str | None = None,
) -> WebSocketCredentials:
    """Load a token and WebSocket URL from ``[environment]`` safely.

    Environment overrides are useful in CI and take precedence over the
    config file:

    * ``AGENTIC_CONFIG_PATH``
    * ``AGENTIC_ENV``
    * ``AGENTIC_ACCESS_TOKEN``
    * ``AGENTIC_WS_URL``
    """
    selected_environment = (environment or os.getenv("AGENTIC_ENV") or DEFAULT_ENVIRONMENT).strip()
    selected_path = Path(
        config_path or os.getenv("AGENTIC_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    ).expanduser()

    token_override = access_token or os.getenv("AGENTIC_ACCESS_TOKEN")
    ws_url_override = ws_url or os.getenv("AGENTIC_WS_URL")

    parser = configparser.ConfigParser()
    if selected_path.exists():
        parser.read(selected_path, encoding="utf-8")
    elif token_override is None or ws_url_override is None:
        raise FileNotFoundError(
            f"Credential file not found: {selected_path}. Set AGENTIC_ACCESS_TOKEN and "
            "AGENTIC_WS_URL to use environment-only credentials."
        )

    if not parser.has_section(selected_environment) and (
        token_override is None or ws_url_override is None
    ):
        raise ValueError(f"Credential section [{selected_environment}] not found in {selected_path}")

    token = _read_config_value(
        parser,
        selected_environment,
        "access_token",
        override=token_override,
    )
    configured_ws_url = _read_config_value(
        parser,
        selected_environment,
        "ws_url",
        override=ws_url_override,
    )
    return WebSocketCredentials(
        access_token=token,
        authenticated_ws_url=build_authenticated_ws_url(configured_ws_url, token),
    )
