import json
import json
import unittest
from unittest.mock import Mock, patch

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from pre_deploy.output import MetricsResults
from custom_benchmarks.tool_calling.models import CanonicalCall, ToolCallObservation


class TestToolCallingEvalDatasetBoundary(unittest.TestCase):
    @staticmethod
    def _observation(scenario_id: str, *, status: str = "completed", completed: bool = True):
        calls = []
        if status == "completed":
            calls = [
                CanonicalCall(
                    ordinal=0,
                    tool_name="websearch",
                    arguments={"query": "ASU"},
                    tool_output={"results": [{"title": "ASU"}]},
                ),
                CanonicalCall(
                    ordinal=1,
                    tool_name="create_artifact",
                    arguments={"prompt": "Use ASU"},
                    tool_output={"success": True},
                ),
            ]
        return ToolCallObservation(
            run_id="run-1",
            scenario_id=scenario_id,
            candidate={"name": "candidate", "provider": "provider"},
            request={"query": "question"},
            status=status,
            completed=completed,
            eos_received=completed,
            tool_calls=calls,
            status_reason="normal completion" if completed else "connection failed",
            final_response="This prose must not be evaluated",
            thinking="private reasoning",
            transport_error=None if completed else "connection failed",
        )

    def test_boundary_uses_canonical_expected_and_observed_trace_text(self):
        scenarios = [
            {
                "scenario_id": "scenario-1",
                "combination_id": "combo-1",
                "variant": 1,
                "questions": ["Find ASU and create an artifact"],
                "enabled_tools": {"websearch": True, "create_artifact": True},
                "expected_tools": ["websearch", "create_artifact"],
                "hop_count": 2,
                "golden_truth": [
                    {"tool_name": "websearch", "arguments": {"query": "ASU"}},
                    {
                        "tool_name": "create_artifact",
                        "arguments": {"prompt": "Use {{steps[0].output}}"},
                    },
                ],
                "tool_catalog": {"websearch": {"required": ["query"]}},
                "system_prompt": "Legacy prompt",
                "metadata": {"construct_type": "multi-hop"},
            }
        ]
        observation = self._observation("scenario-1")

        eval_dataset = benchmark.build_trace_eval_dataset(
            scenarios,
            {"scenario-1": observation},
            dataset_version_id="seed-v1",
        )

        self.assertEqual(len(eval_dataset), 1)
        conversation = eval_dataset.conversations[0]
        expected = json.loads(conversation.metadata["expected_output"])
        actual = json.loads(conversation.messages[1].contents[0].content)

        self.assertEqual(expected["tool_calls"][0]["tool_name"], "websearch")
        self.assertEqual(expected["tool_calls"][1]["arguments"]["prompt"], "Use {{steps[0].output}}")
        self.assertEqual(actual["tool_calls"][0]["tool_output"], {"results": [{"title": "ASU"}]})
        self.assertEqual(actual["tool_calls"][1]["tool_output"], {"success": True})
        self.assertNotIn("final_response", actual)
        self.assertNotIn("thinking", actual)

        self.assertEqual(conversation.metadata["observation_status"], "completed")
        catalog_metadata = json.loads(conversation.metadata["tool_catalog"])
        self.assertEqual(catalog_metadata["websearch"]["name"], "Web Search")
        self.assertEqual(
            catalog_metadata["websearch"]["required_parameters"][0]["name"],
            "query",
        )
        evaluator_catalog = json.loads(conversation.metadata["system_prompt"])
        self.assertEqual(evaluator_catalog, catalog_metadata)
        self.assertEqual(evaluator_catalog["websearch"]["name"], "Web Search")
        self.assertNotIn("Legacy prompt", conversation.metadata["system_prompt"])
        dependencies = json.loads(conversation.metadata["placeholder_dependencies"])
        self.assertEqual(dependencies[0]["step"], 0)
        self.assertEqual(conversation.metadata["source_system_prompt"], "Legacy prompt")

    def test_execution_failures_are_excluded_and_recorded_as_dataset_metadata(self):
        scenarios = [
            {
                "scenario_id": "failed",
                "questions": ["Question"],
                "enabled_tools": {"websearch": True},
                "expected_tools": ["websearch"],
                "hop_count": 1,
                "golden_truth": [{"tool_name": "websearch", "arguments": {"query": "Q"}}],
            },
            {
                "scenario_id": "abstention",
                "questions": ["No tool applies"],
                "enabled_tools": {"websearch": True},
                "expected_tools": [],
                "hop_count": 0,
                "golden_truth": [],
            },
        ]
        observations = {
            "failed": self._observation("failed", status="transport_error", completed=False),
            "abstention": ToolCallObservation(
                run_id="run-1",
                scenario_id="abstention",
                candidate={"name": "candidate", "provider": "provider"},
                request={"query": "No tool applies"},
                status="abstention",
                completed=True,
                eos_received=True,
                status_reason="normal EOS with no calls",
            ),
        }

        eval_dataset = benchmark.build_trace_eval_dataset(scenarios, observations)

        self.assertEqual([conversation.id for conversation in eval_dataset], ["abstention"])
        excluded = json.loads(eval_dataset.metadata["excluded_execution_failures"])
        self.assertEqual(excluded["failed"]["status"], "transport_error")
        actual = json.loads(eval_dataset.conversations[0].messages[1].contents[0].content)
        self.assertEqual(actual, {"tool_calls": []})
    def test_staged_scoring_starts_after_canonical_collection_and_polls_to_completion(self):
        scenario = {
            "scenario_id": "scenario-1",
            "questions": ["Find ASU"],
            "enabled_tools": {"websearch": True},
            "expected_tools": ["websearch"],
            "hop_count": 1,
            "golden_truth": [{"tool_name": "websearch", "arguments": {"query": "ASU"}}],
        }
        observation = self._observation("scenario-1")
        eval_dataset = benchmark.build_trace_eval_dataset(
            [scenario],
            {"scenario-1": observation},
            dataset_version_id="seed-v1",
        )
        lifecycle = []
        completed_result = MetricsResults(
            metrics={},
            dataset_info=eval_dataset.metadata,
            run_id="evaluator-run",
            name="tool_call_failure_modes",
        )

        def start(*, eval_dataset, **kwargs):
            lifecycle.append(("start", eval_dataset))
            self.assertEqual(
                json.loads(eval_dataset.conversations[0].metadata["system_prompt"])["websearch"]["name"],
                "Web Search",
            )

        def finalize(*, eval_dataset, **kwargs):
            lifecycle.append(("finalize", eval_dataset))
            if len([event for event, _ in lifecycle if event == "finalize"]) == 1:
                return {"is_complete": False, "completed_items": 0, "total_items": 1}
            return completed_result

        with patch.object(benchmark, "start_tool_call_failure_modes", side_effect=start) as start_mock, \
             patch.object(benchmark, "finalize_tool_call_failure_modes", side_effect=finalize) as finalize_mock, \
             patch.object(benchmark, "sleep") as sleep_mock:
            result = benchmark.score_accuracy(
                eval_dataset,
                environment=Mock(),
                evaluator_model={"name": "judge", "provider": "provider"},
            )

        self.assertIs(result, completed_result)
        self.assertEqual([event for event, _ in lifecycle], ["start", "finalize", "finalize"])
        self.assertIs(lifecycle[0][1], eval_dataset)
        self.assertEqual(start_mock.call_count, 1)
        self.assertEqual(finalize_mock.call_count, 2)
        sleep_mock.assert_called_once_with(benchmark.POLL_INTERVAL_SECONDS)

    def test_staged_scoring_stops_at_bounded_poll_limit(self):
        eval_dataset = benchmark.build_trace_eval_dataset(
            [{
                "scenario_id": "scenario-1",
                "questions": ["Find ASU"],
                "enabled_tools": {"websearch": True},
                "expected_tools": ["websearch"],
                "hop_count": 1,
                "golden_truth": [{"tool_name": "websearch", "arguments": {"query": "ASU"}}],
            }],
            {"scenario-1": self._observation("scenario-1")},
            dataset_version_id="seed-v1",
        )
        pending = {"is_complete": False, "completed_items": 0, "total_items": 1}
        with patch.object(benchmark, "start_tool_call_failure_modes"), \
             patch.object(
                 benchmark,
                 "finalize_tool_call_failure_modes",
                 return_value=pending,
             ) as finalize_mock, \
             patch.object(benchmark, "MAX_POLLS", 2), \
             patch.object(benchmark, "sleep") as sleep_mock:
            with self.assertRaisesRegex(TimeoutError, "2 polls"):
                benchmark.score_accuracy(
                    eval_dataset,
                    environment=Mock(),
                    evaluator_model={"name": "judge", "provider": "provider"},
                )

        self.assertEqual(finalize_mock.call_count, 2)
        sleep_mock.assert_called_once_with(benchmark.POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    unittest.main()


class TestToolCallingEvaluatorReport(unittest.TestCase):
    @staticmethod
    def _scenario(scenario_id: str, question: str = "Find ASU") -> dict:
        return {
            "scenario_id": scenario_id,
            "combination_id": f"combo-{scenario_id}",
            "variant": 1,
            "questions": [question],
            "enabled_tools": {"websearch": True},
            "expected_tools": ["websearch"],
            "hop_count": 1,
            "golden_truth": [
                {"tool_name": "websearch", "arguments": {"query": "ASU"}}
            ],
        }

    def test_report_maps_results_by_scenario_id_and_retains_evaluator_context(self):
        scenarios = [self._scenario("first"), self._scenario("second")]
        observations = {
            "first": TestToolCallingEvalDatasetBoundary._observation("first"),
            "second": TestToolCallingEvalDatasetBoundary._observation("second"),
        }
        eval_dataset = benchmark.build_trace_eval_dataset(scenarios, observations)
        eval_dataset.metadata.update({
            "evaluator_model": "judge-model",
            "evaluator_provider": "judge-provider",
            "evaluator_run_id": "judge-run-7",
            "metric": "tool_call_failure_modes",
            "metric_name": "tool_call_failure_modes",
            "threshold": 0.5,
            "evaluator_status": "completed",
        })

        class Metric:
            def __init__(self, score, success, reason, dimensions):
                self.score = score
                self.success = success
                self.reason = reason
                self.dimensions = dimensions

        # Deliberately reverse result insertion order: scenario IDs, not
        # positional ordering, define the report/evaluator mapping.
        scored = MetricsResults(
            metrics={
                "second": Metric(0.25, False, "wrong args", {
                    "placeholder_propagation": 0,
                }),
                "first": Metric(0.9, True, "aligned", {
                    "placeholder_propagation": 1,
                }),
            },
            dataset_info=eval_dataset.metadata,
            run_id="judge-run-7",
            name="tool_call_failure_modes",
        )

        rows = benchmark.build_report(
            "candidate-model",
            scored,
            eval_dataset,
            observations=observations,
            scenarios=scenarios,
        )

        self.assertEqual([row["scenario_id"] for row in rows], ["first", "second"])
        self.assertEqual(rows[0]["score"], 0.9)
        self.assertEqual(rows[1]["score"], 0.25)
        self.assertEqual(rows[0]["evaluator_model"], "judge-model")
        self.assertEqual(rows[0]["evaluator_provider"], "judge-provider")
        self.assertEqual(rows[0]["evaluator_run_id"], "judge-run-7")
        self.assertEqual(rows[0]["metric"], "tool_call_failure_modes")
        self.assertEqual(rows[0]["threshold"], 0.5)
        self.assertEqual(rows[0]["placeholder_propagation"], 1)
        self.assertEqual(rows[1]["placeholder_propagation"], 0)
        self.assertEqual(rows[0]["evaluation_status"], "scored")

    def test_detail_row_contains_normalized_trace_and_execution_metadata(self):
        scenario = self._scenario("detail", question="Find ASU")
        scenario["combination_id"] = "combo-detail"
        scenario["variant"] = 3
        observation = TestToolCallingEvalDatasetBoundary._observation("detail")
        eval_dataset = benchmark.build_trace_eval_dataset([scenario], {"detail": observation})

        class Metric:
            score = 0.75
            success = True
            reason = "trace aligned"
            dimensions = {"placeholder_propagation": 1}

        scored = MetricsResults(
            metrics={"detail": Metric()},
            dataset_info=eval_dataset.metadata,
            run_id="judge-run-detail",
            name="tool_call_failure_modes",
        )
        eval_dataset.metadata.update({
            "evaluator_model": "judge-model",
            "evaluator_provider": "judge-provider",
            "evaluator_run_id": "judge-run-detail",
            "evaluator_status": "completed",
            "metric_name": "tool_call_failure_modes",
            "threshold": 0.5,
        })

        row = benchmark.build_report(
            "candidate",
            scored,
            eval_dataset,
            model_provider="candidate-provider",
            observations={"detail": observation},
            scenarios=[scenario],
        )[0]

        self.assertEqual(row["run_id"], "run-1")
        self.assertEqual(row["provider"], "provider")
        self.assertEqual(row["question"], "Find ASU")
        self.assertEqual(row["combination_id"], "combo-detail")
        self.assertEqual(row["variant"], 3)
        self.assertEqual(json.loads(row["expected_calls"])["tool_calls"][0]["tool_name"], "websearch")
        observed = json.loads(row["observed_calls"])
        self.assertEqual(len(observed["tool_calls"]), 2)
        self.assertEqual(json.loads(row["tool_outputs"])[0]["results"][0]["title"], "ASU")
        self.assertEqual(json.loads(row["tool_catalog"])["websearch"]["name"], "Web Search")
        self.assertEqual(json.loads(row["enabled_tools"]), {"websearch": True})
        self.assertEqual(json.loads(row["argument_parse_errors"]), [])
        self.assertEqual(row["observed_call_count"], 2)
        self.assertEqual(row["trace_status"], "completed")
        self.assertEqual(row["status"], "completed")
        self.assertTrue(row["eos_received"])
        self.assertEqual(json.loads(row["errors"]), [])
        self.assertEqual(json.loads(row["request"])["query"], "question")
        self.assertEqual(row["final_response"], "This prose must not be evaluated")
        self.assertEqual(row["thinking"], "private reasoning")
        self.assertEqual(row["placeholder_propagation"], 1)
        self.assertEqual(row["evaluator_model"], "judge-model")
        self.assertEqual(row["evaluation_status"], "scored")
        self.assertEqual(json.loads(row["scenario_metadata"]), {})


        scenarios = [self._scenario("transport"), self._scenario("judge-failure")]
        observations = {
            "transport": TestToolCallingEvalDatasetBoundary._observation(
                "transport", status="transport_error", completed=False
            ),
            "judge-failure": TestToolCallingEvalDatasetBoundary._observation(
                "judge-failure"
            ),
        }
        eval_dataset = benchmark.build_trace_eval_dataset(scenarios, observations)
        rows = benchmark.build_report(
            "candidate-model",
            None,
            eval_dataset,
            observations=observations,
            scenarios=scenarios,
            evaluator_model={"name": "judge", "provider": "provider"},
            evaluator_status="failed",
            evaluator_failure="TimeoutError: evaluator did not finish",
        )

        self.assertEqual([row["scenario_id"] for row in rows], ["transport", "judge-failure"])
        execution_row, evaluation_row = rows
        self.assertEqual(execution_row["evaluation_status"], "not_scored_execution_failure")
        self.assertEqual(execution_row["status"], "transport_error")
        self.assertEqual(execution_row["provider"], "provider")
        self.assertEqual(json.loads(execution_row["expected_calls"])["tool_calls"][0]["tool_name"], "websearch")
        self.assertEqual(json.loads(execution_row["observed_calls"]), {"tool_calls": []})
        self.assertEqual(execution_row["score"], 0.0)
        self.assertFalse(execution_row["success"])
        self.assertEqual(execution_row["evaluator_status"], "not_submitted")
        self.assertEqual(evaluation_row["evaluation_status"], "evaluation_failed")
        self.assertIsNone(evaluation_row["score"])
        self.assertFalse(evaluation_row["success"])
        self.assertEqual(evaluation_row["evaluator_status"], "failed")
        self.assertIn("did not finish", evaluation_row["evaluator_failure"])
        self.assertTrue(evaluation_row["observed_calls"])

    def test_score_accuracy_records_evaluator_identity_and_completion_status(self):
        scenario = self._scenario("scenario-1")
        observation = TestToolCallingEvalDatasetBoundary._observation("scenario-1")
        eval_dataset = benchmark.build_trace_eval_dataset([scenario], {"scenario-1": observation})
        completed_result = MetricsResults(
            metrics={},
            dataset_info=eval_dataset.metadata,
            run_id="evaluator-run",
            name="tool_call_failure_modes",
        )

        with patch.object(benchmark, "start_tool_call_failure_modes"), \
             patch.object(benchmark, "finalize_tool_call_failure_modes", return_value=completed_result):
            result = benchmark.score_accuracy(
                eval_dataset,
                environment=Mock(),
                evaluator_model={"name": "judge", "provider": "provider"},
            )

        self.assertIs(result, completed_result)
        self.assertEqual(eval_dataset.metadata["evaluator_model"], "judge")
        self.assertEqual(eval_dataset.metadata["evaluator_provider"], "provider")
        self.assertEqual(eval_dataset.metadata["evaluator_run_id"], result.run_id)
        self.assertEqual(eval_dataset.metadata["metric_name"], "tool_call_failure_modes")
        self.assertEqual(eval_dataset.metadata["threshold"], benchmark.THRESHOLD)
        self.assertEqual(eval_dataset.metadata["evaluator_status"], "completed")
