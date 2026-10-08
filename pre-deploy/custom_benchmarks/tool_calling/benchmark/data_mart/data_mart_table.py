"""Data-mart table configuration (ported from the pre-release container).

Each data-mart table stores one raw JSON payload per row in a ``SUPER`` column,
with an identity primary key and created/updated timestamps. Downstream views
unnest the ``SUPER`` payload SQL-side, which decouples this writer from upstream
DynamoDB schema changes. These are the same tables other EthAI metrics write to,
so the tool-calling benchmark lands alongside them in the data mart.
"""
from __future__ import annotations

from typing import Any

# Shared data-mart schema for the EthAI metric runs.
_SCHEMA = "aidwnp.ai_data_science"

_SUPER_PAYLOAD_DDL = """
    CREATE TABLE IF NOT EXISTS {table} (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        payload SUPER NOT NULL,
        created_at_dttm TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        last_updated_at_dttm TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    DISTSTYLE AUTO;
"""


class TableConfiguration:
    """Configuration for data-mart tables and their operations."""

    @staticmethod
    def get_table_config(table_name: str) -> dict[str, Any]:
        """Return the main table, columns, and DDL for ``table_name``."""
        configs = {
            "pre_release_ethai_runs": {
                "main_table": f"{_SCHEMA}.pre_release_ethai_runs",
                "primary_key_cols": ["id"],
                "data_cols": ["payload"],
                "super_cols": ["payload"],
                "identity_pk": True,
                "create_main_ddl": _SUPER_PAYLOAD_DDL.format(
                    table=f"{_SCHEMA}.pre_release_ethai_runs"
                ),
            },
            "pre_release_query_processor": {
                "main_table": f"{_SCHEMA}.pre_release_query_processor",
                "primary_key_cols": ["id"],
                "data_cols": ["payload"],
                "super_cols": ["payload"],
                "identity_pk": True,
                "create_main_ddl": _SUPER_PAYLOAD_DDL.format(
                    table=f"{_SCHEMA}.pre_release_query_processor"
                ),
            },
        }

        if table_name not in configs:
            raise ValueError(
                f"{table_name} is not configured. Available tables: {list(configs)}"
            )
        return configs[table_name]
