"""Database I/O: read and write DataFrames from/to a PostgreSQL database."""

from __future__ import annotations

import logging
from typing import Iterable, Tuple

import pandas as pd

from .base import InputSource, OutputManager

logger = logging.getLogger(__name__)


class DatabaseInputSource(InputSource):
    """
    Input source for reading DataFrames from a PostgreSQL database.

    Requires: pip install post-deploy[db]

    Args:
        connection_params: Dict with keys: host, port, database, username, password.
        query: SQL query to execute. If None, reads from schema_name.table_name.
        schema_name: Database schema (used if query is None).
        table_name: Table name (used if query is None).
    """

    def __init__(
        self,
        connection_params: dict[str, str],
        query: str | None = None,
        schema_name: str = "public",
        table_name: str | None = None,
    ):
        self.connection_params = connection_params
        self.query = query
        self.schema_name = schema_name
        self.table_name = table_name

    def load_raw_dataframes(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """Execute the query and yield the result as a single DataFrame."""
        conn = self._get_connection()

        query = self.query
        if query is None:
            if not self.table_name:
                raise ValueError("Either 'query' or 'table_name' must be provided.")
            query = f"SELECT * FROM {self.schema_name}.{self.table_name}"

        logger.info("Executing query: %s", query[:200])

        try:
            df = pd.read_sql(query, conn)
            name = self.table_name or "db_query"
            yield name, df
        finally:
            conn.close()

    def _get_connection(self):
        """Create a psycopg2 connection from connection params."""
        try:
            import psycopg2
        except ImportError:
            raise RuntimeError(
                "psycopg2 is required for database input. "
                "Install with: pip install post-deploy[db]"
            )

        return psycopg2.connect(
            host=self.connection_params["host"],
            port=self.connection_params.get("port", "5432"),
            dbname=self.connection_params["database"],
            user=self.connection_params["username"],
            password=self.connection_params["password"],
        )


class DatabaseOutputManager(OutputManager):
    """
    Output manager that inserts processed DataFrames into a PostgreSQL table.

    Requires: pip install post-deploy[db]

    Args:
        connection_params: Dict with keys: host, port, database, username, password.
        schema_name: Target schema.
        table_name: Target table.
        if_exists: Behavior when table exists ('append', 'replace', 'fail').
    """

    def __init__(
        self,
        connection_params: dict[str, str],
        schema_name: str = "public",
        table_name: str = "metric_output",
        if_exists: str = "append",
    ):
        self.connection_params = connection_params
        self.schema_name = schema_name
        self.table_name = table_name
        self.if_exists = if_exists

    def save(self, name: str, df: pd.DataFrame) -> None:
        """Insert DataFrame rows into the database table."""
        from sqlalchemy import create_engine

        engine = self._get_engine()

        logger.info(
            "Inserting %d rows into %s.%s (source: '%s')",
            len(df), self.schema_name, self.table_name, name,
        )

        df.to_sql(
            name=self.table_name,
            con=engine,
            schema=self.schema_name,
            if_exists=self.if_exists,
            index=False,
        )

        logger.info("Insert complete for '%s'.", name)

    def _get_engine(self):
        """Create a SQLAlchemy engine from connection params."""
        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise RuntimeError(
                "sqlalchemy is required for database output. "
                "Install with: pip install post-deploy[db]"
            )

        p = self.connection_params
        url = f"postgresql://{p['username']}:{p['password']}@{p['host']}:{p.get('port', '5432')}/{p['database']}"
        return create_engine(url)
