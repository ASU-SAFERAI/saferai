"""Staged evaluator lifecycle: start the judge, poll to completion, record status.

Owns ``score_accuracy`` and the evaluator metadata bookkeeping. The module-level
metric functions, ``sleep``, and polling bounds are intentionally rebindable so
tests and the CLI facade can monkeypatch them.
"""
from __future__ import annotations

import uuid
from time import sleep
from typing import Any, Mapping

from pre_deploy.input import EvalDataset
from pre_deploy.metrics.tool_call_failure_modes import (
    finalize_tool_call_failure_modes,
    start_tool_call_failure_modes,
)
from pre_deploy.output import MetricsResults
from pre_deploy.query_processor import AWSEnvironment, RequestDict
from custom_benchmarks.tool_calling.agentic.config import redact_sensitive_text

from ._common import logger
from .config import (
    ASURITE,
    MAX_POLLS,
    POLL_INTERVAL_SECONDS,
    THRESHOLD,
    validate_max_polls,
    validate_poll_interval_seconds,
)


def _set_evaluator_metadata(
    eval_dataset: EvalDataset,
    *,
    evaluator_model: Mapping[str, Any],
    evaluator_run_id: str,
    status: str,
    failure: str | None = None,
) -> None:
    """Persist evaluator identity and lifecycle status beside the dataset.

    ``MetricsResults`` keeps dataset metadata by reference, so this context is
    available both when scoring succeeds and when start/finalize raises.  The
    failure text is already redacted at the boundary and is intentionally
    scalar so it remains compatible with ``EvalDataset`` metadata.
    """
    eval_dataset.metadata.update(
        {
            "evaluator_model": evaluator_model.get("name"),
            "evaluator_provider": evaluator_model.get("provider"),
            "evaluator_run_id": evaluator_run_id,
            "metric": "tool_call_failure_modes",
            "metric_name": "tool_call_failure_modes",
            "threshold": THRESHOLD,
            "evaluator_status": status,
            "evaluator_failure": failure,
        }
    )


def score_accuracy(
    eval_dataset: EvalDataset,
    environment: AWSEnvironment | None,
    evaluator_model: dict,
    *,
    poll_interval_seconds: float | None = None,
    max_polls: int | None = None,
) -> MetricsResults:
    """Stage a tool_call_failure_modes evaluation with a specific judge and poll.

    The judge returns an ``overall_score`` (0-1) plus per-scenario quality
    dimensions (tool_correctness, tool_potentially_valid, json_syntax,
    required_args_present, optional_args_present) where 1 is good and 0 is bad,
    all carried on the returned ``MetricsResults`` and surfaced through its
    ``to_dict()``.  Polling values default to the module constants so existing
    callers and tests that override those constants remain compatible.
    """
    if environment is None:
        environment = AWSEnvironment(target_account_id=None, role_name=None)

    poll_interval_seconds = validate_poll_interval_seconds(
        POLL_INTERVAL_SECONDS if poll_interval_seconds is None else poll_interval_seconds
    )
    max_polls = validate_max_polls(MAX_POLLS if max_polls is None else max_polls)

    evaluator_info = RequestDict(
        asurite=ASURITE,
        run_id=str(uuid.uuid4()),
        model_name=evaluator_model["name"],
        model_provider=evaluator_model["provider"],
        model_temperature=0.5,
    )

    _set_evaluator_metadata(
        eval_dataset,
        evaluator_model=evaluator_model,
        evaluator_run_id=evaluator_info.run_id,
        status="started",
    )

    max_wait_seconds = poll_interval_seconds * max_polls
    logger.info(
        "Starting tool_call_failure_modes (judge=%s/%s, run_id=%s, %d items); "
        "will poll every %.1fs up to %d times (~%.0fs max wait)",
        evaluator_model["name"], evaluator_model["provider"],
        evaluator_info.run_id, len(eval_dataset),
        poll_interval_seconds, max_polls, max_wait_seconds,
    )

    try:
        start_tool_call_failure_modes(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=THRESHOLD,
            environment=environment,
            force_rerun=True,
        )
        logger.info(
            "Enqueued %d judge prompt(s) for run_id=%s; polling for completion",
            len(eval_dataset), evaluator_info.run_id,
        )

        elapsed = 0.0
        for attempt in range(1, max_polls + 1):
            result = finalize_tool_call_failure_modes(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                threshold=THRESHOLD,
                environment=environment,
            )
            if isinstance(result, MetricsResults):
                _set_evaluator_metadata(
                    eval_dataset,
                    evaluator_model=evaluator_model,
                    evaluator_run_id=result.run_id or evaluator_info.run_id,
                    status="completed",
                )
                logger.info(
                    "Scoring complete (run_id=%s) after %d poll(s), ~%.0fs elapsed",
                    evaluator_info.run_id, attempt, elapsed,
                )
                return result
            completed_items = result.get("completed_items")
            total_items = result.get("total_items")
            logger.info(
                "Scoring in progress (poll %d/%d, ~%.0fs elapsed): %s/%s items complete",
                attempt, max_polls, elapsed,
                "?" if completed_items is None else completed_items,
                "?" if total_items is None else total_items,
            )
            if attempt < max_polls:
                sleep(poll_interval_seconds)
                elapsed += poll_interval_seconds

        raise TimeoutError(
            f"tool_call_failure_modes did not finish after {max_polls} polls "
            f"(run_id={evaluator_info.run_id})."
        )
    except Exception as exc:
        _set_evaluator_metadata(
            eval_dataset,
            evaluator_model=evaluator_model,
            evaluator_run_id=evaluator_info.run_id,
            status="failed",
            failure=redact_sensitive_text(f"{type(exc).__name__}: {exc}"),
        )
        raise
