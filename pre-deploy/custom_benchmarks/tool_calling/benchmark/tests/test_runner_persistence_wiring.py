"""Runner-wiring tests for DynamoDB stats persistence.

Verifies the per-model lifecycle placement of the ``ethai_stats`` write:
persistence is on by default (one write per candidate model, keyed by the
evaluator ``run_id``), opt-out via ``--no-write-ddb`` skips the write while
still producing local artifacts, and a write failure propagates (alert-and-
re-raise) without silently succeeding.

The facade copies patched names into the runner namespace for the duration of
``main``, so patching ``custom_benchmarks.run_tool_calling_benchmark`` routes
into ``benchmark.runner``.

Run with::

    python -m pytest custom_benchmarks/tool_calling/benchmark/tests
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from pre_deploy.metrics.tool_call_failure_modes import _METRIC_NAME
from pre_deploy.output import MetricsResults


class _FakeMetric:
    def __init__(self):
        self.name = _METRIC_NAME
        self.score = 1.0
        self.reason = "ok"
        self.success = True
        self.dimensions = {}


def _metrics_results(run_id):
    return MetricsResults(
        metrics={"scenario-1": _FakeMetric()},
        dataset_info={"threshold": 0.5},
        run_id=run_id,
        name=_METRIC_NAME,
    )


class _RunnerHarness:
    """Patch every main() collaborator except the persistence call under test."""

    def __init__(self, candidate_models):
        self.candidate_models = candidate_models
        self._patchers = []

    def __enter__(self):
        def p(name, **kw):
            patcher = patch.object(benchmark, name, **kw)
            self._patchers.append(patcher)
            return patcher.start()

        # The runner lazily constructs these for the stats write; stub them so
        # no STS/boto call is made and persistence can be observed in isolation.
        # They are patched on the facade because ``main`` copies facade globals
        # into the runner namespace (the concern modules re-export
        # ``AWSEnvironment``/``AlertManager``), which would otherwise restore the
        # real classes over a runner-local patch.
        p("AWSEnvironment", return_value=MagicMock(stats_table="ethai_stats_dev"))
        p("AlertManager", return_value=MagicMock())

        # Scenarios + dataset provenance: avoid filesystem/hashing.
        self.load_scenarios = p(
            "load_scenarios",
            return_value=[{"scenario_id": "scenario-1", "questions": ["q"]}],
        )
        p("build_dataset_provenance", return_value={"version": "v1"})
        p("build_registry_provenance", return_value={"version": "r1"})
        p("build_run_manifest", return_value={})

        # Credentials/client are run-level gates; stub them out.
        p("load_credentials", return_value=MagicMock(authenticated_ws_url="wss://x"))
        p("AgenticWebSocketClient", return_value=MagicMock())

        # Candidate selection + execution.
        p("_select_candidate_models", return_value=self.candidate_models)
        p("_resolve_evaluator_model", return_value={"name": "judge", "provider": "jp"})

        # _select_evaluator runs once per model before scoring, so use it to
        # record which model the loop is currently processing.
        def _select_evaluator(model_info, cfg):
            self._current_model = model_info["name"]
            return {"name": "judge", "provider": "jp"}

        p("_select_evaluator", side_effect=_select_evaluator)
        p(
            "collect_candidate_responses",
            return_value={"scenario-1": MagicMock()},
        )
        p("build_trace_eval_dataset", return_value=MagicMock(metadata={}))

        # Scoring returns a MetricsResults carrying a per-model run_id derived
        # from the model the loop is currently processing (set by
        # _select_evaluator, which runs just before scoring).
        def _score(eval_dataset, env, evaluator_model, **kw):
            return _metrics_results(f"run-{self._current_model}")

        p("score_accuracy", side_effect=_score)

        # build_report returns one row carrying the model's run_id so the
        # per-model summary groups correctly.
        def _build_report(model_name, scored, eval_dataset, **kw):
            self._current_model = model_name
            return [
                {
                    "model": model_name,
                    "provider": kw.get("model_provider"),
                    "run_id": scored.run_id if scored is not None else None,
                    "scenario_id": "scenario-1",
                    "score": 1.0,
                    "success": True,
                    "completed": True,
                    "evaluation_status": "scored",
                }
            ]

        # ``score_accuracy`` runs before ``build_report`` but needs the current
        # model name; set it from the model loop via collect side effect.
        self._current_model = self.candidate_models[0]["name"]
        p("build_report", side_effect=_build_report)
        p("write_outputs", return_value={"detail_csv": "d.csv", "summary_csv": "s.csv"})

        # Track persistence calls.
        self.persist = p("persist_model_stats", return_value={})
        return self

    def __exit__(self, *exc):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False


def _run_main(argv):
    with patch("sys.argv", ["run_tool_calling_benchmark.py", *argv]):
        benchmark.main()


class TestRunnerPersistenceWiring(unittest.TestCase):
    def test_writes_one_stats_item_per_model_by_default(self):
        models = [
            {"name": "model-a", "provider": "aws"},
            {"name": "model-b", "provider": "openai"},
        ]
        with _RunnerHarness(models) as harness:
            _run_main([])

            self.assertEqual(harness.persist.call_count, 2)
            run_ids = {c.kwargs["run_id"] for c in harness.persist.call_args_list}
            self.assertEqual(run_ids, {"run-model-a", "run-model-b"})
            # Fixed-contract values are built inside persist_model_stats; the
            # runner passes identity + metrics_results + a summary row.
            first = harness.persist.call_args_list[0]
            self.assertIn("metrics_results", first.kwargs)
            self.assertIn("summary_row", first.kwargs)
            self.assertEqual(first.kwargs["threshold"], 0.5)

    def test_no_write_ddb_skips_persistence_but_still_writes_artifacts(self):
        models = [{"name": "model-a", "provider": "aws"}]
        with _RunnerHarness(models) as harness:
            _run_main(["--no-write-ddb"])

            # Opt-out skips the stats write entirely.
            harness.persist.assert_not_called()
            # Local artifacts are still produced regardless of the DDB opt-out.
            benchmark.write_outputs.assert_called_once()

    def test_persistence_failure_is_isolated_per_model_not_crashing_the_run(self):
        # The writer re-raises on failure, but the runner's per-model
        # try/except isolates it so remaining models still run and local
        # artifacts are written. The failure must be attempted, not skipped.
        models = [
            {"name": "model-a", "provider": "aws"},
            {"name": "model-b", "provider": "openai"},
        ]
        with _RunnerHarness(models) as harness:
            harness.persist.side_effect = RuntimeError("ddb down")
            _run_main([])

            # Both models attempted a write despite the first raising.
            self.assertEqual(harness.persist.call_count, 2)
            # The run still completed and wrote local artifacts.
            benchmark.write_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
