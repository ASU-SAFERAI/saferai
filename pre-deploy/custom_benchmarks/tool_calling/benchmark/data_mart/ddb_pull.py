"""Minimal DynamoDB pull helpers for the data-mart write.

These mirror the sibling ``pre-release`` container's ``query_processor_loader``:
the benchmark has already written a per-model stats item (keyed by ``id =
run_id``) and the staged evaluator has written per-question items (queryable via
the ``run_id_metric_phase`` GSI). To feed the data mart we simply read those
items back by ``run_id`` and hand the raw DynamoDB payloads to the Redshift
writer, which stores them as ``SUPER`` blobs.

The helpers are deliberately small: they resolve the table names and session
from the shared :class:`~pre_deploy.query_processor.AWSEnvironment` and perform a
single ``get_item`` / paginated GSI ``query``. Numeric ``Decimal`` values from
DynamoDB are normalized back to ``int``/``float`` so the payload serializes to
clean JSON for the ``SUPER`` column.
"""
from __future__ import annotations

from typing import Any

from boto3.dynamodb.conditions import Key

from pre_deploy.query_processor import AWSEnvironment
from pre_deploy.query_processor.ddb_utils import format_ddb_data

from .._common import logger

# The staged ``tool_call_failure_modes`` evaluator writes its per-question items
# under this GSI; the pre-release pipeline reads every stage for a run_id.
_RUN_ID_METRIC_PHASE_INDEX = "run_id_metric_phase"


def _table(environment: AWSEnvironment, table_name: str):
    """Resolve a DynamoDB table handle from the shared environment session."""
    dynamodb = environment.session.resource("dynamodb", region_name=environment.region)
    return dynamodb.Table(table_name)


def fetch_stats_results(environment: AWSEnvironment, run_id: str) -> dict[str, Any] | None:
    """Fetch the single stats item for ``run_id`` (PK ``id``), or ``None``.

    Mirrors the pre-release ``fetch_stats_results``: the stats table is keyed
    solely by ``id`` and holds one per-run summary item.
    """
    table_name = environment.stats_table
    logger.debug("Fetching stats item from %s (id=%s)", table_name, run_id)
    response = _table(environment, table_name).get_item(Key={"id": run_id})
    item = response.get("Item")
    if item is None:
        logger.warning("No stats item found in %s for id=%s", table_name, run_id)
        return None
    format_ddb_data(item)
    return item


def fetch_query_processor_results(
    environment: AWSEnvironment,
    run_id: str,
    metric_phase: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch all query_processor items for ``run_id`` via the GSI.

    When ``metric_phase`` is ``None`` every stage for the run is returned,
    matching the pre-release loader. Results are paginated through
    ``LastEvaluatedKey`` and ``Decimal`` values are normalized for JSON.
    """
    table_name = environment.query_processor_table
    table = _table(environment, table_name)

    key_condition = Key("run_id").eq(run_id)
    if metric_phase is not None:
        key_condition &= Key("metric_phase").eq(metric_phase)

    logger.debug(
        "Querying %s GSI %s for run_id=%s (metric_phase=%s)",
        table_name, _RUN_ID_METRIC_PHASE_INDEX, run_id, metric_phase or "<all>",
    )

    items: list[dict[str, Any]] = []
    query_kwargs: dict[str, Any] = {
        "IndexName": _RUN_ID_METRIC_PHASE_INDEX,
        "KeyConditionExpression": key_condition,
    }
    response = table.query(**query_kwargs)
    items.extend(response.get("Items", []))
    while "LastEvaluatedKey" in response:
        query_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        response = table.query(**query_kwargs)
        items.extend(response.get("Items", []))

    for item in items:
        format_ddb_data(item)

    logger.debug(
        "Found %d query_processor item(s) in %s for run_id=%s",
        len(items), table_name, run_id,
    )
    return items
