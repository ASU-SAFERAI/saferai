import math
import unittest

import pandas as pd

import custom_benchmarks.run_tool_calling_benchmark as benchmark


class TestToolCallingSummary(unittest.TestCase):
    def test_summary_uses_explicit_attempted_and_scored_denominators(self):
        report = pd.DataFrame(
            [
                {
                    "model": "candidate",
                    "provider": "provider-a",
                    "run_id": "candidate-run-a",
                    "scenario_id": "scored-pass",
                    "score": 1.0,
                    "success": True,
                    "completed": True,
                    "evaluation_status": "scored",
                    "evaluator_model": "judge",
                    "evaluator_provider": "judge-provider",
                    "evaluator_run_id": "judge-run",
                    "metric": "tool_call_failure_modes",
                    "metric_name": "tool_call_failure_modes",
                    "threshold": 0.5,
                    "placeholder_propagation": 1,
                },
                {
                    "model": "candidate",
                    "provider": "provider-a",
                    "run_id": "candidate-run-a",
                    "scenario_id": "scored-fail",
                    "score": 0.5,
                    "success": False,
                    "completed": True,
                    "evaluation_status": "scored",
                    "evaluator_model": "judge",
                    "evaluator_provider": "judge-provider",
                    "evaluator_run_id": "judge-run",
                    "metric": "tool_call_failure_modes",
                    "metric_name": "tool_call_failure_modes",
                    "threshold": 0.5,
                    "placeholder_propagation": 0,
                },
                {
                    "model": "candidate",
                    "provider": "provider-a",
                    "run_id": "candidate-run-a",
                    "scenario_id": "execution-failure",
                    "score": 0.0,
                    "success": False,
                    "completed": False,
                    "evaluation_status": "not_scored_execution_failure",
                    "evaluator_model": None,
                    "evaluator_provider": None,
                    "evaluator_run_id": None,
                    "metric": "tool_call_failure_modes",
                    "metric_name": "tool_call_failure_modes",
                    "threshold": 0.5,
                    "placeholder_propagation": None,
                },
                {
                    "model": "candidate",
                    "provider": "provider-a",
                    "run_id": "candidate-run-a",
                    "scenario_id": "evaluation-failure",
                    "score": None,
                    "success": False,
                    "completed": True,
                    "evaluation_status": "evaluation_failed",
                    "evaluator_model": "judge",
                    "evaluator_provider": "judge-provider",
                    "evaluator_run_id": "judge-run",
                    "metric": "tool_call_failure_modes",
                    "metric_name": "tool_call_failure_modes",
                    "threshold": 0.5,
                    "placeholder_propagation": None,
                },
                {
                    "model": "candidate",
                    "provider": "provider-b",
                    "run_id": "candidate-run-b",
                    "scenario_id": "other-provider",
                    "score": 0.75,
                    "success": True,
                    "completed": True,
                    "evaluation_status": "scored",
                    "evaluator_model": "judge-b",
                    "evaluator_provider": "judge-provider-b",
                    "evaluator_run_id": "judge-run-b",
                    "metric": "tool_call_failure_modes",
                    "metric_name": "tool_call_failure_modes",
                    "threshold": 0.5,
                    "placeholder_propagation": 1,
                },
            ]
        )

        summary = benchmark.build_summary(report)

        self.assertEqual(
            list(summary[["model", "provider", "run_id"]].itertuples(index=False, name=None)),
            [
                ("candidate", "provider-b", "candidate-run-b"),
                ("candidate", "provider-a", "candidate-run-a"),
            ],
        )
        provider_a = summary.loc[summary["provider"] == "provider-a"].iloc[0]
        self.assertEqual(provider_a["questions"], 4)
        self.assertEqual(provider_a["scored_questions"], 3)
        self.assertEqual(provider_a["completed"], 3)
        self.assertEqual(provider_a["execution_failures"], 1)
        self.assertEqual(provider_a["evaluation_failures"], 1)
        self.assertAlmostEqual(provider_a["mean_score"], 0.5)
        # One passing row out of four attempts; the null evaluator score is
        # not silently removed from the pass-rate denominator.
        self.assertAlmostEqual(provider_a["pass_rate"], 0.25)
        self.assertEqual(provider_a["threshold"], 0.5)
        self.assertEqual(provider_a["evaluator_run_id"], "judge-run")
        self.assertAlmostEqual(provider_a["placeholder_propagation_rate"], 0.5)

    def test_summary_keeps_zero_scores_scored_and_reports_no_score_mean_as_nan(self):
        report = pd.DataFrame(
            [
                {
                    "model": "candidate",
                    "provider": "provider",
                    "run_id": "run",
                    "scenario_id": "zero-score",
                    "score": 0.0,
                    "success": False,
                    "completed": True,
                    "evaluation_status": "scored",
                    "threshold": 0.5,
                },
                {
                    "model": "candidate-without-score",
                    "provider": "provider",
                    "run_id": "run-no-score",
                    "scenario_id": "evaluation-failure",
                    "score": None,
                    "success": False,
                    "completed": True,
                    "evaluation_status": "evaluation_failed",
                    "threshold": 0.5,
                },
            ]
        )

        summary = benchmark.build_summary(report)

        zero_score = summary.loc[summary["model"] == "candidate"].iloc[0]
        self.assertEqual(zero_score["questions"], 1)
        self.assertEqual(zero_score["scored_questions"], 1)
        self.assertEqual(zero_score["mean_score"], 0.0)
        self.assertEqual(zero_score["pass_rate"], 0.0)

        no_score = summary.loc[
            summary["model"] == "candidate-without-score"
        ].iloc[0]
        self.assertEqual(no_score["questions"], 1)
        self.assertEqual(no_score["scored_questions"], 0)
        self.assertTrue(math.isnan(no_score["mean_score"]))
        self.assertEqual(no_score["pass_rate"], 0.0)
        self.assertEqual(no_score["evaluation_failures"], 1)


if __name__ == "__main__":
    unittest.main()
