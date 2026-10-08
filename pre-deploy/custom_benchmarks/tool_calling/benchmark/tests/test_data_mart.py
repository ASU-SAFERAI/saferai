"""Tests for the Redshift data-mart write path (``benchmark.data_mart``).

Everything is mocked: psycopg2 connections, the DynamoDB session/table, and the
credential source. These verify the ported pre-release behavior without a live
Redshift or DynamoDB dependency:

* table configuration (SUPER payload schema, main table names),
* the append-only INSERT SQL (JSON_PARSE wrapping, columns, timestamps),
* _ensure_table runs the DDL once per process and rollback on insert failure,
* the DynamoDB pull helpers (get_item by id, GSI query with pagination),
* flatten_results_for_scenario reshaping,
* the writer orchestration (pull -> serialize -> insert per table),
* data-mart credential precedence (env over conf) and the psycopg2 lazy import.

Run with::

    python -m pytest custom_benchmarks/tool_calling/benchmark/tests
"""
from __future__ import annotations

import json
import sys
import unittest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from custom_benchmarks.tool_calling.benchmark.data_mart import (
    data_mart_manager,
    writer,
)
from custom_benchmarks.tool_calling.benchmark.data_mart.config import (
    DataMartCredentials,
    load_data_mart_credentials,
)
from custom_benchmarks.tool_calling.benchmark.data_mart.data_mart_manager import (
    DatabaseManager,
    InsertManager,
)
from custom_benchmarks.tool_calling.benchmark.data_mart.data_mart_table import (
    TableConfiguration,
)
from custom_benchmarks.tool_calling.benchmark.data_mart.ddb_pull import (
    fetch_query_processor_results,
    fetch_stats_results,
)


def _credentials():
    return DataMartCredentials(
        username="u",
        password="p",
        host="h",
        database="aidwnp",
        port="5439",
    )


class TestTableConfiguration(unittest.TestCase):
    def test_known_tables_use_super_payload_schema(self):
        for name in ("pre_release_ethai_runs", "pre_release_query_processor"):
            config = TableConfiguration.get_table_config(name)
            self.assertEqual(config["main_table"], f"aidwnp.ai_data_science.{name}")
            self.assertEqual(config["data_cols"], ["payload"])
            self.assertEqual(config["super_cols"], ["payload"])
            self.assertIn("CREATE TABLE IF NOT EXISTS", config["create_main_ddl"])
            self.assertIn("payload SUPER NOT NULL", config["create_main_ddl"])
            self.assertIn("IDENTITY(1,1)", config["create_main_ddl"])

    def test_unknown_table_rejected(self):
        with self.assertRaises(ValueError):
            TableConfiguration.get_table_config("not_a_table")


class TestInsertSql(unittest.TestCase):
    def test_super_column_wrapped_in_json_parse(self):
        config = TableConfiguration.get_table_config("pre_release_ethai_runs")
        sql = InsertManager._build_insert_sql(config)
        self.assertIn("INSERT INTO aidwnp.ai_data_science.pre_release_ethai_runs", sql)
        self.assertIn("(payload, created_at_dttm, last_updated_at_dttm)", sql)
        self.assertIn("VALUES (JSON_PARSE(%s), CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)", sql)


class _FakeConnection:
    """Context-manager connection whose cursor records executed statements."""

    def __init__(self, recorder, fail_on_executemany=False):
        self._recorder = recorder
        self._fail = fail_on_executemany
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        conn = self

        class _Cursor:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def execute(self_inner, sql, params=None):
                conn._recorder.setdefault("execute", []).append((sql, params))

            def executemany(self_inner, sql, seq):
                conn._recorder.setdefault("executemany", []).append((sql, list(seq)))
                if conn._fail:
                    raise RuntimeError("insert boom")

        return _Cursor()

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class TestInsertManager(unittest.TestCase):
    def _manager(self, recorder, fail=False):
        db = MagicMock(spec=DatabaseManager)
        connections = []

        def _get_connection():
            conn = _FakeConnection(recorder, fail_on_executemany=fail)
            connections.append(conn)
            return conn

        db.get_connection.side_effect = _get_connection
        return InsertManager(db), connections

    def test_ensure_table_runs_ddl_once_then_inserts(self):
        recorder: dict = {}
        manager, connections = self._manager(recorder)

        manager.insert_records([("{\"a\":1}",)], "pre_release_ethai_runs")
        manager.insert_records([("{\"a\":2}",)], "pre_release_ethai_runs")

        # DDL executed exactly once across two inserts (ensured per process).
        create_statements = [
            sql for sql, _ in recorder.get("execute", [])
            if "CREATE TABLE IF NOT EXISTS" in sql
        ]
        self.assertEqual(len(create_statements), 1)
        self.assertEqual(len(recorder.get("executemany", [])), 2)

    def test_insert_failure_rolls_back_and_reraises(self):
        recorder: dict = {}
        manager, connections = self._manager(recorder, fail=True)

        with self.assertRaises(RuntimeError):
            manager.insert_records([("{\"a\":1}",)], "pre_release_query_processor")

        # The connection used for the failing insert was rolled back.
        self.assertTrue(any(c.rolled_back for c in connections))


class TestDatabaseManagerLazyDriver(unittest.TestCase):
    def test_connection_params_shape(self):
        db = DatabaseManager(_credentials())
        self.assertEqual(db.connection_params["dbname"], "aidwnp")
        self.assertEqual(db.connection_params["sslmode"], "require")
        self.assertEqual(db.connection_params["port"], "5439")

    def test_missing_psycopg2_raises_actionable_error(self):
        # Simulate psycopg2 being absent at connection time.
        with patch.dict(sys.modules, {"psycopg2": None}):
            db = DatabaseManager(_credentials())
            with self.assertRaises(ImportError) as ctx:
                db.get_connection()
            self.assertIn("psycopg2", str(ctx.exception))

    def test_get_connection_uses_psycopg2_connect(self):
        fake_psycopg2 = MagicMock()
        with patch.dict(sys.modules, {"psycopg2": fake_psycopg2}):
            db = DatabaseManager(_credentials())
            db.get_connection()
            fake_psycopg2.connect.assert_called_once()
            kwargs = fake_psycopg2.connect.call_args.kwargs
            self.assertEqual(kwargs["sslmode"], "require")


class TestDdbPull(unittest.TestCase):
    def _environment(self):
        env = MagicMock()
        env.stats_table = "ethai_stats_dev"
        env.query_processor_table = "query_processor_dev"
        env.region = "us-west-2"
        table = MagicMock()
        env.session.resource.return_value.Table.return_value = table
        return env, table

    def test_fetch_stats_returns_item_and_normalizes_decimals(self):
        env, table = self._environment()
        table.get_item.return_value = {"Item": {"id": "run-1", "threshold": Decimal("0.5")}}
        item = fetch_stats_results(env, "run-1")
        self.assertEqual(item["id"], "run-1")
        # Decimal normalized back to a JSON-friendly number.
        self.assertEqual(item["threshold"], 0.5)
        table.get_item.assert_called_once_with(Key={"id": "run-1"})

    def test_fetch_stats_missing_returns_none(self):
        env, table = self._environment()
        table.get_item.return_value = {}
        self.assertIsNone(fetch_stats_results(env, "absent"))

    def test_fetch_query_processor_paginates_gsi(self):
        env, table = self._environment()
        table.query.side_effect = [
            {"Items": [{"id": "a"}], "LastEvaluatedKey": {"id": "a"}},
            {"Items": [{"id": "b"}]},
        ]
        items = fetch_query_processor_results(env, "run-1")
        self.assertEqual([i["id"] for i in items], ["a", "b"])
        first_kwargs = table.query.call_args_list[0].kwargs
        self.assertEqual(first_kwargs["IndexName"], "run_id_metric_phase")
        # Second call carries the pagination token.
        self.assertIn("ExclusiveStartKey", table.query.call_args_list[1].kwargs)


class TestFlattenResults(unittest.TestCase):
    def test_flattens_tool_calling_results_map(self):
        item = {
            "scenario_name": "thinking_loop_tool_calling",
            "metrics": {
                "results": {
                    "scenario-1": {"score": 1.0, "success": True},
                    "scenario-2": {"score": 0.0, "success": False},
                },
                "summary": {"mean_score": 0.5},
            },
        }
        out = writer.flatten_results_for_scenario(item)
        self.assertIsInstance(out["results"], list)
        numbers = {row["question_number"] for row in out["results"]}
        self.assertEqual(numbers, {"scenario-1", "scenario-2"})
        # results removed from metrics to avoid duplication; summary retained.
        self.assertNotIn("results", out["metrics"])
        self.assertIn("summary", out["metrics"])

    def test_non_target_scenario_unchanged(self):
        item = {"scenario_name": "OTHER", "metrics": {"results": {"x": {}}}}
        out = writer.flatten_results_for_scenario(item)
        self.assertNotIn("results", out)
        self.assertIn("results", out["metrics"])


class TestWriterOrchestration(unittest.TestCase):
    def test_pulls_then_inserts_super_payloads_per_table(self):
        env = MagicMock()
        insert_manager = MagicMock()

        stats_item = {
            "id": "run-1",
            "scenario_name": "thinking_loop_tool_calling",
            "metrics": {"results": {"scenario-1": {"score": 1.0}}},
        }
        qp_items = [{"id": "qp-1", "run_id": "run-1"}, {"id": "qp-2", "run_id": "run-1"}]

        with patch.object(writer, "fetch_stats_results", return_value=stats_item), \
             patch.object(writer, "fetch_query_processor_results", return_value=qp_items):
            counts = writer.write_benchmark_run_to_data_mart(
                ["run-1"],
                environment=env,
                insert_manager=insert_manager,
            )

        self.assertEqual(counts, {"pre_release_ethai_runs": 1, "pre_release_query_processor": 2})
        # Two insert calls: one per table, each with SUPER-payload tuples.
        tables_written = [c.args[1] for c in insert_manager.insert_records.call_args_list]
        self.assertEqual(
            sorted(tables_written),
            ["pre_release_ethai_runs", "pre_release_query_processor"],
        )
        stats_call = next(
            c for c in insert_manager.insert_records.call_args_list
            if c.args[1] == "pre_release_ethai_runs"
        )
        payload = json.loads(stats_call.args[0][0][0])
        # Flatten applied: results is now a top-level list.
        self.assertIsInstance(payload["results"], list)

    def test_no_run_ids_skips_write(self):
        insert_manager = MagicMock()
        counts = writer.write_benchmark_run_to_data_mart(
            [], environment=MagicMock(), insert_manager=insert_manager
        )
        insert_manager.insert_records.assert_not_called()
        self.assertEqual(counts["pre_release_ethai_runs"], 0)

    def test_missing_stats_item_still_writes_query_processor(self):
        env = MagicMock()
        insert_manager = MagicMock()
        with patch.object(writer, "fetch_stats_results", return_value=None), \
             patch.object(writer, "fetch_query_processor_results", return_value=[{"id": "qp-1"}]):
            counts = writer.write_benchmark_run_to_data_mart(
                ["run-1"], environment=env, insert_manager=insert_manager
            )
        self.assertEqual(counts["pre_release_ethai_runs"], 0)
        self.assertEqual(counts["pre_release_query_processor"], 1)


class TestCredentialLoading(unittest.TestCase):
    def test_env_overrides_take_precedence(self):
        env = {
            "DATA_MART_USERNAME": "env_user",
            "DATA_MART_PASSWORD": "env_pass",
            "DATA_MART_HOST": "env_host",
            "DATA_MART_DATABASE": "aidwnp",
            "DATA_MART_PORT": "5439",
        }
        with patch.dict("os.environ", env, clear=False):
            creds = load_data_mart_credentials(config_path="/nonexistent.conf")
        self.assertEqual(creds.username, "env_user")
        self.assertEqual(creds.host, "env_host")
        self.assertEqual(creds.environment, "NONPROD")

    def test_password_redacted_in_repr(self):
        creds = _credentials()
        self.assertIn("<redacted>", repr(creds))
        self.assertNotIn("p", repr(creds).split("password=")[1][:12])

    def test_unknown_environment_rejected(self):
        with self.assertRaises(ValueError):
            load_data_mart_credentials(environment="STAGING")


if __name__ == "__main__":
    unittest.main()
