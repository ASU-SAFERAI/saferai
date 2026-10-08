"""Candidate execution: run scenarios through the agentic client into observations.

Owns the per-scenario execution loop, retry/concurrency handling, and the
conversion of transport/construction failures into explicit observations so
every attempted scenario is recorded before scoring.
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from custom_benchmarks.tool_calling.agentic.client import (
    AgenticWebSocketClient,
    WebSocketQueryResult,
)
from custom_benchmarks.tool_calling.agentic.config import (
    redact_sensitive_data,
    redact_sensitive_text,
)
from custom_benchmarks.tool_calling.agentic.observation import build_tool_call_observation
from custom_benchmarks.tool_calling.agentic.protocol import StreamTrace
from custom_benchmarks.tool_calling.agentic.runner import build_agentic_request
from custom_benchmarks.tool_calling.models import ToolCallObservation

from ._common import logger
from .config import DEFAULT_CONCURRENCY, DEFAULT_RETRY_COUNT, build_execution_policy


def _failure_observation(
    *,
    run_id: str,
    scenario_id: str,
    candidate: dict,
    request: dict,
    expected_call_count: int | None,
    error: Exception,
    phase: str,
    include_raw_frames: bool,
) -> ToolCallObservation:
    """Create an explicit observation when one scenario cannot execute.

    The normal WebSocket client already converts transport failures into a
    ``WebSocketQueryResult``.  This boundary also handles client doubles,
    request construction, and normalization failures so every attempted
    scenario has a record before scoring starts.
    """
    message = redact_sensitive_text(f"{phase}: {type(error).__name__}: {error}")
    trace = StreamTrace()
    trace.mark_transport_error(message)
    return build_tool_call_observation(
        WebSocketQueryResult(request=request, trace=trace),
        run_id=run_id,
        scenario_id=scenario_id,
        candidate=candidate,
        expected_call_count=expected_call_count,
        include_raw_frames=include_raw_frames,
    )


def _annotate_execution(
    observation: ToolCallObservation,
    *,
    policy: dict[str, Any],
    attempt_statuses: list[str],
) -> ToolCallObservation:
    """Attach policy and bounded attempt diagnostics to an observation."""
    observation.stream_summary = {
        **observation.stream_summary,
        "attempt_count": len(attempt_statuses),
        "attempt_statuses": list(attempt_statuses),
        "execution_policy": dict(policy),
    }
    return observation


def _collect_one_scenario(
    model_info: dict,
    scenario: dict,
    client: AgenticWebSocketClient,
    *,
    candidate_run_id: str,
    include_raw_frames: bool,
    policy: dict[str, Any],
    retry_count: int,
    client_factory: Callable[[], AgenticWebSocketClient] | None,
) -> tuple[str, ToolCallObservation]:
    """Execute one scenario, using a fresh client for concurrent attempts."""
    scenario_id = scenario["scenario_id"]
    question = scenario["questions"][0].strip()
    request: dict = {}
    try:
        request = build_agentic_request(
            question,
            scenario["enabled_tools"],
            model_name=model_info["name"],
            model_provider=model_info["provider"],
        )
    except Exception as exc:
        observation = _failure_observation(
            run_id=candidate_run_id,
            scenario_id=scenario_id,
            candidate=model_info,
            request=request,
            expected_call_count=scenario.get("hop_count"),
            error=exc,
            phase="agentic request construction failed",
            include_raw_frames=include_raw_frames,
        )
        return scenario_id, _annotate_execution(
            observation,
            policy=policy,
            attempt_statuses=[observation.status],
        )

    attempt_statuses: list[str] = []
    for attempt in range(retry_count + 1):
        logger.info(
            "Collecting agentic response for candidate %s/%s scenario=%s "
            "attempt=%d/%d tools=%s",
            model_info["name"],
            model_info["provider"],
            scenario_id,
            attempt + 1,
            retry_count + 1,
            request["agent_config"]["tools"],
        )
        try:
            # Concurrent execution never shares the caller's client.  The
            # factory must create a new client, and AgenticWebSocketClient
            # itself opens one socket per query and keeps no query state.
            query_client = client_factory() if client_factory is not None else client
            result = query_client.query(request)
            observation = build_tool_call_observation(
                result,
                run_id=candidate_run_id,
                scenario_id=scenario_id,
                candidate=model_info,
                expected_call_count=scenario.get("hop_count"),
                include_raw_frames=include_raw_frames,
            )
        except Exception as exc:
            observation = _failure_observation(
                run_id=candidate_run_id,
                scenario_id=scenario_id,
                candidate=model_info,
                request=request,
                expected_call_count=scenario.get("hop_count"),
                error=exc,
                phase="agentic scenario execution failed",
                include_raw_frames=include_raw_frames,
            )
            logger.error(
                "Agentic execution failed for candidate %s/%s scenario=%s "
                "attempt=%d; continuing: %s",
                model_info.get("name"),
                model_info.get("provider"),
                scenario_id,
                attempt + 1,
                redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
            )

        attempt_statuses.append(observation.status)
        if observation.completed or attempt == retry_count:
            logger.info(
                "Collected candidate %s/%s scenario=%s status=%s completed=%s "
                "observed_calls=%d attempts=%d",
                model_info["name"],
                model_info["provider"],
                scenario_id,
                observation.status,
                observation.completed,
                len(observation.tool_calls),
                len(attempt_statuses),
            )
            return scenario_id, _annotate_execution(
                observation,
                policy=policy,
                attempt_statuses=attempt_statuses,
            )
        logger.warning(
            "Retrying candidate %s/%s scenario=%s after status=%s; "
            "retry policy is explicitly enabled",
            model_info.get("name"),
            model_info.get("provider"),
            scenario_id,
            observation.status,
        )

    raise AssertionError("bounded execution loop completed without an observation")


def collect_candidate_responses(
    model_info: dict,
    scenarios: list[dict],
    client: AgenticWebSocketClient,
    *,
    run_id: str | None = None,
    include_raw_frames: bool = False,
    concurrency: int = DEFAULT_CONCURRENCY,
    retry_count: int = DEFAULT_RETRY_COUNT,
    client_factory: Callable[[], AgenticWebSocketClient] | None = None,
) -> dict[str, ToolCallObservation]:
    """Collect observations sequentially by default or with bounded workers.

    The default path remains source-order sequential and uses the supplied
    client, preserving existing callers and side-effect safety.  When
    concurrency is greater than one, a client factory is required so each
    scenario/attempt gets an independent client and socket.  Results are
    inserted in source order even though worker completion order may differ.
    """
    if not isinstance(include_raw_frames, bool):
        raise TypeError("include_raw_frames must be a boolean")
    policy = build_execution_policy(
        concurrency=concurrency,
        retry_count=retry_count,
    )
    if concurrency > 1 and client_factory is None:
        raise ValueError(
            "client_factory is required when concurrency is greater than one "
            "to prevent sharing a WebSocket client"
        )

    candidate_run_id = run_id or str(uuid.uuid4())
    scenario_list = list(scenarios)
    total = len(scenario_list)
    logger.info(
        "Collecting %d scenario(s) for candidate %s/%s (mode=%s, concurrency=%d, run_id=%s)",
        total,
        model_info.get("name"),
        model_info.get("provider"),
        policy["mode"],
        concurrency,
        candidate_run_id,
    )
    if concurrency == 1:
        completed = []
        for index, scenario in enumerate(scenario_list, start=1):
            logger.info(
                "Scenario %d/%d (%s) for candidate %s/%s",
                index,
                total,
                scenario.get("scenario_id"),
                model_info.get("name"),
                model_info.get("provider"),
            )
            completed.append(
                _collect_one_scenario(
                    model_info,
                    scenario,
                    client,
                    candidate_run_id=candidate_run_id,
                    include_raw_frames=include_raw_frames,
                    policy=policy,
                    retry_count=retry_count,
                    client_factory=None,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    _collect_one_scenario,
                    model_info,
                    scenario,
                    client,
                    candidate_run_id=candidate_run_id,
                    include_raw_frames=include_raw_frames,
                    policy=policy,
                    retry_count=retry_count,
                    client_factory=client_factory,
                )
                for scenario in scenario_list
            ]
            # Log worker completions as they finish so a concurrent run still
            # shows forward progress, then read results in submission order to
            # keep the returned mapping deterministic.
            finished = 0
            for _ in as_completed(futures):
                finished += 1
                logger.info(
                    "Completed %d/%d scenario worker(s) for candidate %s/%s",
                    finished,
                    total,
                    model_info.get("name"),
                    model_info.get("provider"),
                )
            completed = [future.result() for future in futures]

    responses = {scenario_id: observation for scenario_id, observation in completed}
    completed_count = sum(1 for obs in responses.values() if obs.completed)
    logger.info(
        "Finished collecting scenarios for candidate %s/%s: %d/%d completed successfully",
        model_info.get("name"),
        model_info.get("provider"),
        completed_count,
        total,
    )
    return responses


def _serialize_candidate_responses(
    responses: dict[str, ToolCallObservation],
) -> dict[str, str]:
    """Serialize normalized observations for the existing evaluator boundary."""
    return {
        scenario_id: json.dumps(
            redact_sensitive_data(observation.to_dict()),
            ensure_ascii=False,
            sort_keys=True,
        )
        for scenario_id, observation in responses.items()
    }
