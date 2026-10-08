"""Orchestrate the DynamoDB pull and the Redshift data-mart write.

Mirrors the sibling ``pre-release`` container's ``main``: for each benchmark
``run_id`` (one per candidate model) it reads the stats item and the
query_processor stage items back from DynamoDB, serializes each as a single
JSON ``SUPER`` payload, and appends them to the shared data-mart tables:

* stats  -> ``aidwnp.ai_data_science.pre_release_ethai_runs``
* QP rows -> ``aidwnp.ai_data_science.pre_release_query_processor``

The write is append-only. ``flatten_results_for_scenario`` reshapes the nested
per-question ``metrics.results`` map into a list for the stats payload, matching
how MARBLE/CARE rows are flattened in the pre-release pipeline (the tool-calling
``results`` map is keyed by scenario id, so the same shape applies).
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from pre_deploy.query_processor import AWSEnvironment

from .._common import logger
from .config import DataMartCredentials, load_data_mart_credentials
from .data_mart_manager import DatabaseManager, InsertManager
from .ddb_pull import fetch_query_processor_results, fetch_stats_results

STATS_TABLE = "pre_release_ethai_runs"
QUERY_PROCESSOR_TABLE = "pre_release_query_processor"

# The tool-calling stats item uses this scenario_name; its nested results map is
# keyed by scenario id and benefits from the same flatten the pre-release
# pipeline applies to MARBLE/CARE.
_FLATTEN_SCENARIOS = {"thinking_loop_tool_calling", "MARBLE", "CARE"}


def flatten_results_for_scenario(item: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten a stats item's ``metrics.results`` map into a top-level list.

    Converts ``{"scenario-1": {"score": ...}, ...}`` into
    ``[{"question_number": "scenario-1", "score": ...}, ...]`` placed at the
    top-level ``results`` key, and removes ``metrics.results`` to avoid
    duplication. Items for other scenarios, or without a dict results map, are
    returned unchanged (as a shallow copy).
    """
    scenario_name = item.get("scenario_name", "")
    if scenario_name not in _FLATTEN_SCENARIOS:
        return dict(item)

    metrics = item.get("metrics")
    if not isinstance(metrics, Mapping):
        return dict(item)

    results = metrics.get("results")
    if not isinstance(results, Mapping):
        return dict(item)

    flattened = [
        {"question_number": key, **(value if isinstance(value, Mapping) else {"value": value})}
        for key, value in results.items()
    ]

    new_item = dict(item)
    new_item["results"] = flattened
    new_item["metrics"] = {k: v for k, v in metrics.items() if k != "results"}
    return new_item


def _super_payload(value: Any) -> tuple[str]:
    """Serialize one record as a single-column ``SUPER`` insert tuple."""
    return (json.dumps(value, default=str, ensure_ascii=False),)


def write_benchmark_run_to_data_mart(
    run_ids: Iterable[str],
    *,
    environment: AWSEnvironment | None = None,
    credentials: DataMartCredentials | None = None,
    config_path: str | None = None,
    data_mart_environment: str | None = None,
    insert_manager: InsertManager | None = None,
) -> dict[str, int]:
    """Pull each run's items from DynamoDB and append them to the data mart.

    Returns a count of rows written per table. ``environment`` (DynamoDB) and
    ``insert_manager``/``credentials`` (Redshift) are injectable so tests can
    supply mocks; by default they are constructed from ambient AWS credentials
    and the resolved data-mart credentials.
    """
    run_id_list = [rid for rid in run_ids if rid]
    if not run_id_list:
        logger.info("No run_ids supplied; skipping data-mart write.")
        return {STATS_TABLE: 0, QUERY_PROCESSOR_TABLE: 0}

    if environment is None:
        environment = AWSEnvironment(target_account_id=None, role_name=None)

    if insert_manager is None:
        if credentials is None:
            credentials = load_data_mart_credentials(
                config_path=config_path,
                environment=data_mart_environment,
            )
        insert_manager = InsertManager(DatabaseManager(credentials))
        logger.info(
            "Writing to the %s data mart (%s)",
            credentials.environment, credentials.host,
        )

    # Step 1: pull stats items (one per run_id) and flatten their results.
    stats_records: list[tuple[str]] = []
    for run_id in run_id_list:
        item = fetch_stats_results(environment, run_id)
        if item is not None:
            stats_records.append(_super_payload(flatten_results_for_scenario(item)))

    if stats_records:
        insert_manager.insert_records(stats_records, STATS_TABLE)
        logger.info("Persisted %d stats record(s) to Redshift", len(stats_records))
    else:
        logger.info("No stats records found to persist")

    # Step 2: pull all query_processor stage items (every stage per run_id).
    qp_records: list[tuple[str]] = []
    for run_id in run_id_list:
        qp_records.extend(
            _super_payload(item)
            for item in fetch_query_processor_results(environment, run_id)
        )

    if qp_records:
        insert_manager.insert_records(qp_records, QUERY_PROCESSOR_TABLE)
        logger.info("Persisted %d query_processor record(s) to Redshift", len(qp_records))
    else:
        logger.info("No query_processor records found to persist")

    return {
        STATS_TABLE: len(stats_records),
        QUERY_PROCESSOR_TABLE: len(qp_records),
    }
