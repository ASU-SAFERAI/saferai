"""DynamoDB persistence scaffold for the tool-calling benchmark.

The staged ``tool_call_failure_modes`` evaluator already writes its per-question
judge items into the ``query_processor`` table during its ``eval`` phase, keyed
by a per-item ``id`` and queryable through the ``run_id_metric_phase`` GSI under
the model's ``run_id``. This module adds the only net-new write: one summary
item per candidate model into the ``ethai_stats`` table, keyed by partition key
``id = run_id`` with no sort key.

The stats writer is ported here rather than imported from the sibling engine
repository (which is not a dependency of this project). It reuses the vendored
``pre_deploy.query_processor`` primitives:

* :class:`~pre_deploy.query_processor.AWSEnvironment` for table-name resolution
  (``stats_table``, default ``ethai_stats_dev``), region, and the boto3 session.
* :class:`~pre_deploy.query_processor.alert_manager.AlertManager` for the SNS
  error-notification failure path.
* :func:`~pre_deploy.query_processor.ddb_utils.format_data_for_ddb` for the
  ``float`` -> :class:`~decimal.Decimal` conversion DynamoDB requires.

``metric_name`` and ``metric_phase`` are sourced from the evaluator module
constants so the persisted values cannot drift from the metric that produced
them. ``scenario_name``, ``evaluate``, and ``metric_type`` are fixed per the
tool-calling benchmark's agreed persistence contract.

Nothing in this module reads ``varun_credentials.conf``; AWS access uses the
ambient same-account credentials resolved by ``AWSEnvironment``.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

from pre_deploy.metrics.tool_call_failure_modes import _METRIC_NAME, _PHASE
from pre_deploy.output import MetricsResults
from pre_deploy.query_processor import AWSEnvironment
from pre_deploy.query_processor.alert_manager import AlertManager
from pre_deploy.query_processor.ddb_utils import format_data_for_ddb

from ._common import logger

# Fixed persistence contract for the tool-calling stats item. These values are
# intentionally literal (not derived from the agentic websocket run): the stats
# table rows describe the tool-calling benchmark as a single logical scenario.
STATS_ASURITE = "ethai-datascience"
STATS_SCENARIO_NAME = "thinking_loop_tool_calling"


def build_stats_item(
    *,
    run_id: str,
    model_name: str,
    model_provider: str,
    evaluator_model_name: str | None,
    evaluator_model_provider: str | None,
    threshold: float,
    metrics_results: MetricsResults | None,
    summary_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape one ``ethai_stats`` summary item for a candidate model.

    The item is keyed by ``id = run_id`` (the evaluator ``run_id`` carried on
    the model's :class:`MetricsResults`), with no sort key, matching the
    existing stats-table schema. ``metric_name`` and ``metric_phase`` come from
    the ``tool_call_failure_modes`` module constants so they cannot diverge from
    the evaluator. The ``metrics`` map embeds the per-scenario judge results and,
    when available, the per-model summary aggregates.

    This builder constructs the dict directly rather than reading fields off a
    ``RequestDict``: this repository's ``RequestDict`` does not carry the
    stats-specific fields the sibling engine's own request object exposes.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")

    metrics: dict[str, Any] = {}
    if metrics_results is not None:
        # ``to_dict()["results"]`` is {scenario_id: {score, reason, success,
        # dimensions}} â€” the per-scenario judge verdict keyed by scenario ID.
        metrics["results"] = metrics_results.to_dict().get("results", {})
    if summary_row is not None:
        metrics["summary"] = _summary_payload(summary_row)

    return {
        # Partition key (no sort key).
        "id": run_id,
        "asurite": STATS_ASURITE,
        "scenario_name": STATS_SCENARIO_NAME,
        "evaluate": None,
        "metric_type": None,
        # Sourced from the evaluator module so they cannot drift.
        "metric_name": _METRIC_NAME,
        "metric_phase": _PHASE,
        # Candidate and evaluator identity.
        "model_name": model_name,
        "model_provider": model_provider,
        "evaluator_model_name": evaluator_model_name,
        "evaluator_model_provider": evaluator_model_provider,
        "threshold": threshold,
        "request_end_timestamp": int(time.time()),
        "metrics": metrics,
    }


def _summary_payload(summary_row: Mapping[str, Any]) -> dict[str, Any]:
    """Project the per-model summary row into the stats ``metrics.summary`` map.

    Only the denominator-explicit aggregate fields are embedded; identity
    fields (model/provider/run_id/evaluator) already live at the top level of
    the stats item. Missing keys are simply omitted so the payload stays
    compatible with older/custom summary frames.
    """
    keys = (
        "questions",
        "scored_questions",
        "completed",
        "execution_failures",
        "evaluation_failures",
        "mean_score",
        "pass_rate",
    )
    return {key: summary_row[key] for key in keys if key in summary_row}


def write_stats_item(
    environment: AWSEnvironment,
    alert_manager: AlertManager,
    stats_item: Mapping[str, Any],
) -> None:
    """Write one summary item to the ``ethai_stats`` table.

    Floats are converted to :class:`~decimal.Decimal` via ``format_data_for_ddb``
    before ``put_item`` because DynamoDB rejects native floats. On failure the
    error is reported through :meth:`AlertManager.notify_error` with the write
    context and the ``run_id``/``id``, then re-raised, mirroring the existing
    ``write_to_query_processor_table`` convention so a persistence failure is
    neither hidden nor silently swallowed.
    """
    table_name = environment.stats_table
    item = format_data_for_ddb(dict(stats_item))
    try:
        session = environment.session
        dynamodb = session.resource("dynamodb", region_name=environment.region)
        table = dynamodb.Table(table_name)
        logger.debug("Writing stats item to %s (id=%s)", table_name, stats_item.get("id"))
        table.put_item(Item=item)
        logger.info(
            "Wrote stats item to %s (id=%s)", table_name, stats_item.get("id")
        )
    except Exception as exc:
        logger.error(
            "Failed to write stats item to %s (id=%s): %s",
            table_name,
            stats_item.get("id"),
            exc,
        )
        alert_manager.notify_error(
            context="write_to_stats_table",
            exception=exc,
            context_data={"table_name": table_name, "id": stats_item.get("id")},
            log_level="ERROR",
        )
        raise


def persist_model_stats(
    environment: AWSEnvironment,
    alert_manager: AlertManager,
    *,
    run_id: str,
    model_name: str,
    model_provider: str,
    evaluator_model_name: str | None,
    evaluator_model_provider: str | None,
    threshold: float,
    metrics_results: MetricsResults | None,
    summary_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and write a candidate model's stats item; return the written item.

    Convenience wrapper used by the runner so the per-model lifecycle has a
    single call site. The returned item is the pre-serialization dict (useful
    for the manifest and tests); the Decimal conversion happens inside
    :func:`write_stats_item` only for the DynamoDB payload.
    """
    stats_item = build_stats_item(
        run_id=run_id,
        model_name=model_name,
        model_provider=model_provider,
        evaluator_model_name=evaluator_model_name,
        evaluator_model_provider=evaluator_model_provider,
        threshold=threshold,
        metrics_results=metrics_results,
        summary_row=summary_row,
    )
    write_stats_item(environment, alert_manager, stats_item)
    return stats_item
