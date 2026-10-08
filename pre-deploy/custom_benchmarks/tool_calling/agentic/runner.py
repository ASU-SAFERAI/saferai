#!/usr/bin/env python
"""Run agentic WebSocket probes for one or more supported tools.

Examples:
    python -m custom_benchmarks.tool_calling.agentic.runner --dry-run
    python -m custom_benchmarks.tool_calling.agentic.runner --all-tools-together
    python -m custom_benchmarks.tool_calling.agentic.runner --output results/agentic_tools.json

Live mode reads [DEV] from varun_credentials.conf by default. Prefer
AGENTIC_ACCESS_TOKEN and AGENTIC_WS_URL environment variables in CI; neither
credentials nor the authenticated URL is printed or written to the report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Support direct execution from the repository root.
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

if __package__ in {None, ""}:
    from custom_benchmarks.tool_calling.agentic.client import (
        AgenticWebSocketClient,
        WebSocketQueryResult,
    )
    from custom_benchmarks.tool_calling.agentic.config import (
        DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
        load_credentials,
        redact_sensitive_text,
    )
else:
    from .client import AgenticWebSocketClient, WebSocketQueryResult
    from .config import (
        DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
        load_credentials,
        redact_sensitive_text,
    )

from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY

if __package__ in {None, ""}:
    from custom_benchmarks.tool_calling.benchmark._common import configure_logging, logger
else:
    from ..benchmark._common import configure_logging, logger

# The registry is the source of truth for both validation and request ordering.
# Keep this alias for CLI compatibility and callers of the existing probe runner.
SUPPORTED_TOOLS = tuple(TOOL_REGISTRY)


def _build_default_tool_query(tool_name: str, definition: Mapping[str, Any]) -> str:
    """Build a probe query from the registry description for one tool."""
    description = str(definition.get("description", "")).strip().rstrip(".")
    if description:
        description = description[0].lower() + description[1:]
    return f"Use {tool_name} to {description}."


# Probe defaults are projected from the live registry so newly registered tools
# cannot be omitted from the default request path.
DEFAULT_TOOL_QUERIES = {
    tool_name: _build_default_tool_query(tool_name, definition)
    for tool_name, definition in TOOL_REGISTRY.items()
}


def _build_default_multi_tool_query() -> str:
    """Build a registry-complete query for the combined probe."""
    tools = ", ".join(TOOL_REGISTRY)
    return (
        "Complete this coordinated tool-calling task by using every registered "
        f"tool exactly once and then summarize the results: {tools}."
    )


DEFAULT_MULTI_TOOL_QUERY = _build_default_multi_tool_query()


def _validate_tools(tools: Iterable[str]) -> tuple[str, ...]:
    """Validate tool IDs and return them in canonical registry order."""
    selected = tuple(tools)
    if not selected:
        raise ValueError("At least one tool must be selected")
    if any(not isinstance(tool, str) or not tool for tool in selected):
        raise ValueError("Tool names must be non-empty strings")

    unknown = sorted(set(selected) - set(TOOL_REGISTRY))
    if unknown:
        raise ValueError(f"Unsupported tool(s): {', '.join(unknown)}")

    selected_set = set(selected)
    return tuple(tool for tool in TOOL_REGISTRY if tool in selected_set)


def _enabled_tool_names(
    enabled_tools: Mapping[str, bool] | Iterable[str],
) -> tuple[str, ...]:
    """Normalize a scenario map or tool-name iterable for an agent request."""
    if isinstance(enabled_tools, Mapping):
        selected = tuple(
            tool_name
            for tool_name, enabled in enabled_tools.items()
            if enabled is True
        )
    else:
        if isinstance(enabled_tools, (str, bytes)):
            raise TypeError("enabled_tools must be a mapping or iterable of tool names")
        selected = tuple(enabled_tools)
    return _validate_tools(selected)


def build_agentic_request(
    question: str,
    enabled_tools: Mapping[str, bool] | Iterable[str],
    *,
    model_provider: str = "openai",
    model_name: str = "gpt4o",
) -> dict[str, Any]:
    """Build the agentic WebSocket request for one candidate/scenario.

    ``enabled_tools`` may be the dataset's boolean tool map or an iterable of
    tool IDs. In both cases, the returned ``agent_config.tools`` list follows
    ``TOOL_REGISTRY`` insertion order. The agents endpoint owns its system
    prompt and tool schemas, so this adapter sends the question verbatim and
    intentionally has no ``system_prompt`` field.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must not be empty")
    if not isinstance(model_provider, str) or not model_provider.strip():
        raise ValueError("model_provider must not be empty")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must not be empty")

    selected_tools = _enabled_tool_names(enabled_tools)
    return {
        "action": "queryV2",
        "query": question,
        "model_provider": model_provider,
        "model_name": model_name,
        "agentic": True,
        "agent_config": {"tools": list(selected_tools)},
        "response_format": {"type": "json"},
    }


def build_query_payload(
    tool_name: str,
    query: str,
    *,
    model_provider: str = "openai",
    model_name: str = "gpt4o",
) -> dict[str, Any]:
    """Build a queryV2 agentic payload for one tool.

    Kept as a compatibility wrapper around :func:`build_agentic_request`.
    """
    return build_agentic_request(
        query,
        (tool_name,),
        model_provider=model_provider,
        model_name=model_name,
    )


def build_multi_tool_payload(
    tools: tuple[str, ...] = SUPPORTED_TOOLS,
    query: str = DEFAULT_MULTI_TOOL_QUERY,
    *,
    model_provider: str = "openai",
    model_name: str = "gpt4o",
) -> dict[str, Any]:
    """Build one request that enables all selected tools for one query."""
    return build_agentic_request(
        query,
        tools,
        model_provider=model_provider,
        model_name=model_name,
    )


def _log_stream_details(tool_name: str, result: WebSocketQueryResult) -> None:
    """Log compact stream summaries rather than one line per frame."""
    stream = result.trace
    logger.info(
        "tool=%s stream_frames=%d event_counts=%s",
        tool_name,
        len(stream.frames),
        stream.event_counts,
    )
    for call in stream.compact_tool_calls():
        output = call.get("output", {})
        logger.info(
            "tool=%s tool_call name=%s id=%s events=%d output_fields=%s",
            tool_name,
            call.get("tool_name"),
            call.get("tool_call_id"),
            call.get("event_count"),
            sorted(output),
        )
    if stream.eos_metadata:
        logger.info("tool=%s EOS metadata=%s", tool_name, stream.eos_metadata)
    if stream.transport_error:
        logger.error("tool=%s transport error=%s", tool_name, stream.transport_error)
    if stream.errors:
        logger.error("tool=%s stream errors=%d", tool_name, len(stream.errors))
    if stream.response:
        logger.info("tool=%s response_chars=%d", tool_name, len(stream.response))
    if stream.thinking:
        logger.info("tool=%s thinking_chars=%d", tool_name, len(stream.thinking))


def _execute_probe(
    client: AgenticWebSocketClient,
    request: dict[str, Any],
    *,
    report_key: str,
    include_raw_stream: bool,
) -> dict[str, dict[str, Any]]:
    logger.info(
        "sending probe=%s action=%s model=%s/%s tools=%s query=%s",
        report_key,
        request["action"],
        request["model_provider"],
        request["model_name"],
        request["agent_config"]["tools"],
        request["query"],
    )
    result: WebSocketQueryResult = client.query(request)
    stream = result.trace
    logger.info(
        "probe=%s eos=%s completed=%s logical_tool_calls=%d errors=%d",
        report_key,
        stream.eos_received,
        stream.completed,
        len(stream.compact_tool_calls()),
        len(stream.errors),
    )
    _log_stream_details(report_key, result)
    return {report_key: result.to_dict(include_raw_stream=include_raw_stream)}


def run_tool_probes(
    client: AgenticWebSocketClient,
    *,
    tools: tuple[str, ...] = SUPPORTED_TOOLS,
    queries: dict[str, str] | None = None,
    include_raw_stream: bool = False,
    all_tools_together: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run per-tool probes or one combined multi-tool probe."""
    selected_tools = _validate_tools(tools)
    if all_tools_together:
        request = build_multi_tool_payload(selected_tools)
        return _execute_probe(
            client,
            request,
            report_key="all_tools_together",
            include_raw_stream=include_raw_stream,
        )

    selected_queries = queries or DEFAULT_TOOL_QUERIES
    results: dict[str, dict[str, Any]] = {}
    for tool_name in selected_tools:
        request = build_query_payload(tool_name, selected_queries[tool_name])
        results.update(
            _execute_probe(
                client,
                request,
                report_key=tool_name,
                include_raw_stream=include_raw_stream,
            )
        )
    return results


def _dry_run_report(
    tools: tuple[str, ...],
    *,
    all_tools_together: bool = False,
) -> dict[str, Any]:
    if all_tools_together:
        return {
            "mode": "dry-run-multi-tool",
            "tools": {
                "all_tools_together": {
                    "request": build_multi_tool_payload(tools),
                }
            },
        }
    return {
        "mode": "dry-run",
        "tools": {
            tool: {
                "request": build_query_payload(tool, DEFAULT_TOOL_QUERIES[tool]),
            }
            for tool in tools
        },
    }


def _log_dry_run_report(report: dict[str, Any]) -> None:
    """Log dry-run requests in a readable form rather than dumping JSON."""
    logger.info("dry run: no WebSocket connection opened")
    for probe_name, details in report["tools"].items():
        request = details["request"]
        logger.info(
            "dry-run probe=%s action=%s model=%s/%s tools=%s query=%s",
            probe_name,
            request["action"],
            request["model_provider"],
            request["model_name"],
            request["agent_config"]["tools"],
            request["query"],
        )


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote probe report to %s", output_path)


def _parse_tools(raw_tools: list[str] | None) -> tuple[str, ...]:
    return _validate_tools(tuple(raw_tools or SUPPORTED_TOOLS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--environment", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--timeout", type=float, default=DEFAULT_WEBSOCKET_TIMEOUT_SECONDS)
    parser.add_argument("--tool", action="append", dest="tools", choices=SUPPORTED_TOOLS)
    parser.add_argument(
        "--all-tools-together",
        action="store_true",
        help="Send one query with all selected tools enabled instead of one query per tool",
    )
    parser.add_argument(
        "--include-raw-stream",
        action="store_true",
        help="Include every raw frame and normalized tool event in the JSON report",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print payloads without opening a WebSocket")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    try:
        tools = _parse_tools(args.tools)
        if args.dry_run:
            report = _dry_run_report(tools, all_tools_together=args.all_tools_together)
        else:
            credentials = load_credentials(config_path=args.config_path, environment=args.environment)
            client = AgenticWebSocketClient(credentials, timeout=args.timeout)
            report = {
                "mode": "live-multi-tool" if args.all_tools_together else "live",
                "environment": args.environment,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "tools": run_tool_probes(
                    client,
                    tools=tools,
                    include_raw_stream=args.include_raw_stream,
                    all_tools_together=args.all_tools_together,
                ),
            }
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        parser.error(redact_sensitive_text(str(exc)))

    if args.output:
        write_report(report, args.output)
    elif args.dry_run:
        _log_dry_run_report(report)
    else:
        logger.info("live probe run complete; inspect the compact logs above")


if __name__ == "__main__":
    main()
