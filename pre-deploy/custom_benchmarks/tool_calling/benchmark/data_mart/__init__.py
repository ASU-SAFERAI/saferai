"""Redshift data-mart persistence for the tool-calling benchmark.

Mirrors the sibling ``pre-release`` container's data-mart pattern: after the
benchmark writes its per-model summary to the ``ethai_stats`` DynamoDB table
(and the staged evaluator writes per-question items to ``query_processor``),
this package *pulls those items back from DynamoDB by run_id* and writes them as
raw ``SUPER`` payloads into the shared data-mart tables other EthAI metrics use:

* ``aidwnp.ai_data_science.pre_release_ethai_runs``
* ``aidwnp.ai_data_science.pre_release_query_processor``

The DynamoDB pull is intentionally minimal (``ddb_pull``); the Redshift write
reuses the append-only ``id IDENTITY + payload SUPER`` schema and ``JSON_PARSE``
insert used across the data mart (``data_mart_manager`` / ``data_mart_table``).
Credentials come from the ``DATA_MART_*`` config sections or Secrets Manager
(``config``); nothing here logs secret-bearing values.
"""

from .data_mart_manager import DatabaseManager, InsertManager
from .data_mart_table import TableConfiguration
from .ddb_pull import fetch_query_processor_results, fetch_stats_results
from .writer import flatten_results_for_scenario, write_benchmark_run_to_data_mart

__all__ = [
    "DatabaseManager",
    "InsertManager",
    "TableConfiguration",
    "fetch_stats_results",
    "fetch_query_processor_results",
    "flatten_results_for_scenario",
    "write_benchmark_run_to_data_mart",
]
