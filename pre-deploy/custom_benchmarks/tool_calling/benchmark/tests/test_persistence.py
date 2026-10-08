"""Tests for the DynamoDB persistence scaffold (``benchmark.persistence``).

These exercise the net-new ``ethai_stats`` write path in isolation. The staged
``tool_call_failure_modes`` evaluator owns the ``query_processor`` writes, so
these tests also assert that the persistence module introduces no second
query_processor write path.

Run with::

    python -m pytest custom_benchmarks/tool_calling/benchmark/tests
"""
from __future__ import annotations

import unittest
from decimal import Decimal
from unittest.mock import MagicMock

from pre_deploy.metrics.tool_call_failure_modes import _METRIC_NAME, _PHASE
from pre_deploy.output import MetricsResults

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from custom_benchmarks.tool_calling.benchmark import persistence


class _FakeMetric:
    """Minimal metric object compatible with ``MetricsResults.to_dict``."""

    def __init__(self, score, reason, success, dimensions):
        self.name = _METRIC_NAME
        self.score = score
        self.reason = reason
        self.success = success
        self.dimensions = dimensions


def _metrics_results(run_id="run-123"):
    """A real MetricsResults with one fake scored scenario."""
    metric = _FakeMetric(
        score=1.0,
        reason="matched golden trace",
        success=True,
        dimensions={
            "tool_correctness": 1,
            "tool_potentially_valid": 1,
            "json_syntax": 1,
            "required_args_present": 1,
            "optional_args_present": 1,
            "placeholder_propagation": 1,
        },
    )
    return MetricsResults(
        metrics={"scenario-1": metric},
        dataset_info={"threshold": 0.5},
        run_id=run_id,
        name=_METRIC_NAME,
    )


class TestBuildStatsItem(unittest.TestCase):
    def _build(self, **overrides):
        kwargs = dict(
            run_id="run-123",
            model_name="voxtrol-mini",
            model_provider="aws",
            evaluator_model_name="gemma4_31b_it",
            evaluator_model_provider="asu-air",
            threshold=0.5,
            metrics_results=_metrics_results(),
            summary_row={
                "model": "voxtrol-mini",
                "provider": "aws",
                "run_id": "run-123",
                "questions": 3,
                "scored_questions": 2,
                "completed": 2,
                "execution_failures": 1,
                "evaluation_failures": 0,
                "mean_score": 0.75,
                "pass_rate": 0.3333,
            },
        )
        kwargs.update(overrides)
        return persistence.build_stats_item(**kwargs)

    def test_primary_key_is_run_id_with_no_sort_key(self):
        item = self._build()
        self.assertEqual(item["id"], "run-123")
        # The stats table is keyed solely by ``id``; there must be no sort key.
        self.assertNotIn("sort_key", item)
        self.assertNotIn("metric_phase_sort", item)

    def test_fixed_contract_fields(self):
        item = self._build()
        self.assertEqual(item["asurite"], "ethai-datascience")
        self.assertEqual(item["scenario_name"], "thinking_loop_tool_calling")
        self.assertIsNone(item["evaluate"])
        self.assertIsNone(item["metric_type"])

    def test_metric_name_and_phase_sourced_from_constants(self):
        item = self._build()
        self.assertEqual(item["metric_name"], _METRIC_NAME)
        self.assertEqual(item["metric_phase"], _PHASE)
        # Guard against a literal that could silently drift from the evaluator.
        self.assertEqual(item["metric_name"], "tool_call_failure_modes")
        self.assertEqual(item["metric_phase"], "eval")

    def test_candidate_and_evaluator_identity_preserved(self):
        item = self._build()
        self.assertEqual(item["model_name"], "voxtrol-mini")
        self.assertEqual(item["model_provider"], "aws")
        self.assertEqual(item["evaluator_model_name"], "gemma4_31b_it")
        self.assertEqual(item["evaluator_model_provider"], "asu-air")
        self.assertEqual(item["threshold"], 0.5)

    def test_request_end_timestamp_is_epoch_int(self):
        item = self._build()
        self.assertIsInstance(item["request_end_timestamp"], int)
        self.assertGreater(item["request_end_timestamp"], 0)

    def test_metrics_embeds_results_and_summary(self):
        item = self._build()
        results = item["metrics"]["results"]
        self.assertIn("scenario-1", results)
        self.assertEqual(results["scenario-1"]["score"], 1.0)
        self.assertEqual(results["scenario-1"]["success"], True)
        self.assertIn("dimensions", results["scenario-1"])

        summary = item["metrics"]["summary"]
        # Identity fields stay at the top level; only aggregates are embedded.
        self.assertEqual(summary["questions"], 3)
        self.assertEqual(summary["scored_questions"], 2)
        self.assertEqual(summary["mean_score"], 0.75)
        self.assertNotIn("model", summary)
        self.assertNotIn("run_id", summary)

    def test_summary_optional(self):
        item = self._build(summary_row=None)
        self.assertNotIn("summary", item["metrics"])
        self.assertIn("results", item["metrics"])

    def test_metrics_results_optional(self):
        item = self._build(metrics_results=None, summary_row=None)
        self.assertEqual(item["metrics"], {})

    def test_empty_run_id_rejected(self):
        with self.assertRaises(ValueError):
            self._build(run_id="")


class TestWriteStatsItem(unittest.TestCase):
    def _environment(self):
        env = MagicMock()
        env.stats_table = "ethai_stats_dev"
        env.region = "us-west-2"
        table = MagicMock()
        env.session.resource.return_value.Table.return_value = table
        return env, table

    def test_float_converted_to_decimal_before_put(self):
        env, table = self._environment()
        alert_manager = MagicMock()
        stats_item = {
            "id": "run-123",
            "threshold": 0.5,
            "metrics": {"summary": {"mean_score": 0.75}},
        }

        persistence.write_stats_item(env, alert_manager, stats_item)

        table.put_item.assert_called_once()
        written = table.put_item.call_args.kwargs["Item"]
        # DynamoDB rejects native floats; the writer must convert them.
        self.assertIsInstance(written["threshold"], Decimal)
        self.assertIsInstance(written["metrics"]["summary"]["mean_score"], Decimal)
        alert_manager.notify_error.assert_not_called()

    def test_targets_resolved_stats_table(self):
        env, table = self._environment()
        persistence.write_stats_item(env, MagicMock(), {"id": "run-123"})
        env.session.resource.assert_called_once_with(
            "dynamodb", region_name="us-west-2"
        )
        env.session.resource.return_value.Table.assert_called_once_with(
            "ethai_stats_dev"
        )

    def test_failure_alerts_and_reraises(self):
        env, table = self._environment()
        alert_manager = MagicMock()
        boom = RuntimeError("ddb unavailable")
        table.put_item.side_effect = boom

        with self.assertRaises(RuntimeError):
            persistence.write_stats_item(env, alert_manager, {"id": "run-123"})

        alert_manager.notify_error.assert_called_once()
        call = alert_manager.notify_error.call_args
        self.assertEqual(call.kwargs["context"], "write_to_stats_table")
        self.assertEqual(call.kwargs["context_data"]["id"], "run-123")
        self.assertEqual(
            call.kwargs["context_data"]["table_name"], "ethai_stats_dev"
        )
        self.assertIs(call.kwargs["exception"], boom)


class TestPersistModelStats(unittest.TestCase):
    def test_builds_and_writes_returning_item(self):
        env = MagicMock()
        env.stats_table = "ethai_stats_dev"
        env.region = "us-west-2"
        table = MagicMock()
        env.session.resource.return_value.Table.return_value = table
        alert_manager = MagicMock()

        item = persistence.persist_model_stats(
            env,
            alert_manager,
            run_id="run-xyz",
            model_name="gpt5_6_luna",
            model_provider="openai",
            evaluator_model_name="gemma4_31b_it",
            evaluator_model_provider="asu-air",
            threshold=0.5,
            metrics_results=_metrics_results(run_id="run-xyz"),
            summary_row=None,
        )

        self.assertEqual(item["id"], "run-xyz")
        self.assertEqual(item["model_name"], "gpt5_6_luna")
        table.put_item.assert_called_once()


class TestNoSecondQueryProcessorPath(unittest.TestCase):
    """The evaluator owns query_processor writes; persistence adds only stats."""

    def test_persistence_module_has_no_query_processor_writer(self):
        public = {name for name in dir(persistence) if not name.startswith("_")}
        self.assertNotIn("write_to_query_processor_table", public)
        self.assertFalse(
            any("query_processor" in name for name in public),
            msg="persistence must not introduce a query_processor write path",
        )

    def test_facade_exposes_stats_writer(self):
        # The facade re-exports persistence helpers so patches route through.
        self.assertTrue(hasattr(benchmark, "persist_model_stats"))
        self.assertTrue(hasattr(benchmark, "build_stats_item"))
        self.assertTrue(hasattr(benchmark, "write_stats_item"))


if __name__ == "__main__":
    unittest.main()
