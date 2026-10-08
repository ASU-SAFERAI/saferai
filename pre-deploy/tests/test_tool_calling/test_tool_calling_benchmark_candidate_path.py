import inspect
import json
import unittest
from unittest.mock import Mock

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from custom_benchmarks.tool_calling.agentic.client import WebSocketQueryResult
from custom_benchmarks.tool_calling.agentic.protocol import StreamTrace
from custom_benchmarks.tool_calling.models import ToolCallObservation


class TestToolCallingBenchmarkCandidatePath(unittest.TestCase):
    @staticmethod
    def _websocket_result(tool_name: str) -> WebSocketQueryResult:
        trace = StreamTrace()
        trace.add(json.dumps({
            "tool_call": {
                "type": "function",
                "name": tool_name,
                "tool_call_id": f"call-{tool_name}",
                "function": {"arguments": '{"query":"find"}'},
            }
        }))
        trace.add(json.dumps({"response": "<EOS>"}))
        return WebSocketQueryResult(
            request={"query": "find"},
            trace=trace,
        )

    def test_collects_agentic_requests_without_legacy_system_prompt(self):
        websocket_result = self._websocket_result("websearch")
        client = Mock()
        client.query.return_value = websocket_result
        scenarios = [
            {
                "scenario_id": "scenario-1",
                "questions": ["Use the enabled search tool."],
                "system_prompt": "Legacy prompt must never be sent to the agent endpoint.",
                "enabled_tools": {
                    "create_artifact": True,
                    "websearch": True,
                    "generate_image": False,
                },
            }
        ]
        model = {"name": "candidate-model", "provider": "candidate-provider"}

        responses = benchmark.collect_candidate_responses(model, scenarios, client)

        self.assertIsInstance(responses["scenario-1"], ToolCallObservation)
        observation = responses["scenario-1"]
        self.assertEqual(observation.status, "completed")
        self.assertEqual(observation.tool_calls[0].tool_name, "websearch")
        self.assertEqual(observation.tool_calls[0].arguments, {"query": "find"})
        client.query.assert_called_once()
        request = client.query.call_args.args[0]
        self.assertEqual(request["query"], scenarios[0]["questions"][0])
        self.assertEqual(request["model_name"], model["name"])
        self.assertEqual(request["model_provider"], model["provider"])
        self.assertEqual(request["agent_config"]["tools"], ["websearch", "create_artifact"])
        self.assertNotIn("system_prompt", request)
        self.assertNotIn("system_prompt", request["agent_config"])
        self.assertNotIn(scenarios[0]["system_prompt"], json.dumps(request))

    def test_executes_scenarios_in_source_order_and_builds_observations(self):
        scenarios = [
            {
                "scenario_id": "scenario-first",
                "questions": ["First question"],
                "enabled_tools": {"websearch": True},
                "hop_count": 1,
            },
            {
                "scenario_id": "scenario-second",
                "questions": ["Second question"],
                "enabled_tools": {"create_artifact": True},
                "hop_count": 0,
            },
        ]
        client = Mock()
        client.query.side_effect = [
            self._websocket_result("websearch"),
            self._websocket_result("create_artifact"),
        ]
        model = {"name": "candidate-model", "provider": "candidate-provider"}

        observations = benchmark.collect_candidate_responses(
            model,
            scenarios,
            client,
            run_id="candidate-run",
        )

        self.assertEqual(list(observations), ["scenario-first", "scenario-second"])
        self.assertEqual(
            [call.args[0]["query"] for call in client.query.call_args_list],
            ["First question", "Second question"],
        )
        self.assertEqual(observations["scenario-first"].run_id, "candidate-run")
        self.assertEqual(observations["scenario-second"].status, "completed")
        self.assertEqual(observations["scenario-second"].tool_calls[0].tool_name, "create_artifact")

    def test_failed_scenario_is_recorded_and_later_scenarios_continue(self):
        client = Mock()
        client.query.side_effect = [
            RuntimeError("connection failed with Bearer secret-token"),
            self._websocket_result("create_artifact"),
        ]
        scenarios = [
            {
                "scenario_id": "scenario-failed",
                "questions": ["First question"],
                "enabled_tools": {"websearch": True},
                "hop_count": 1,
            },
            {
                "scenario_id": "scenario-after-failure",
                "questions": ["Second question"],
                "enabled_tools": {"create_artifact": True},
                "hop_count": 1,
            },
        ]

        observations = benchmark.collect_candidate_responses(
            {"name": "candidate-model", "provider": "candidate-provider"},
            scenarios,
            client,
            run_id="candidate-run",
        )

        self.assertEqual(list(observations), ["scenario-failed", "scenario-after-failure"])
        failed = observations["scenario-failed"]
        self.assertEqual(failed.status, "transport_error")
        self.assertFalse(failed.completed)
        self.assertIn("connection failed", failed.transport_error)
        self.assertNotIn("secret-token", failed.transport_error)
        self.assertEqual(observations["scenario-after-failure"].status, "completed")
        self.assertEqual(client.query.call_count, 2)
        serialized = json.loads(benchmark._serialize_candidate_responses(observations)["scenario-failed"])
        self.assertEqual(serialized["status"], "transport_error")
        self.assertIn("transport_error", serialized)

    def test_benchmark_source_has_no_query_processor_candidate_generation(self):
        source = inspect.getsource(benchmark)

        self.assertNotIn("DeepEvalClient", source)
        self.assertNotIn("batch_generate", source)
        self.assertNotIn("_build_query", source)
        self.assertIn("AgenticWebSocketClient", source)
        self.assertIn("client.query(request)", source)

    def test_serializes_normalized_observations_for_existing_evaluator_boundary(self):
        observation = self._websocket_result("websearch")
        normalized = benchmark.collect_candidate_responses(
            {"name": "model", "provider": "provider"},
            [{
                "scenario_id": "scenario-1",
                "questions": ["find"],
                "enabled_tools": {"websearch": True},
                "hop_count": 1,
            }],
            Mock(query=Mock(return_value=observation)),
            run_id="run-1",
        )["scenario-1"]

        serialized = benchmark._serialize_candidate_responses({"scenario-1": normalized})

        payload = json.loads(serialized["scenario-1"])
        self.assertEqual(payload["scenario_id"], "scenario-1")
        self.assertEqual(payload["tool_calls"][0]["tool_name"], "websearch")
        self.assertEqual(payload["tool_calls"][0]["arguments"], {"query": "find"})

    @staticmethod
    def _transport_result() -> WebSocketQueryResult:
        trace = StreamTrace()
        trace.mark_transport_error("connection failed")
        return WebSocketQueryResult(request={"query": "failed"}, trace=trace)

    @staticmethod
    def _scenario(scenario_id: str, question: str) -> dict:
        return {
            "scenario_id": scenario_id,
            "questions": [question],
            "enabled_tools": {"websearch": True},
            "hop_count": 1,
        }

    def test_default_policy_is_sequential_and_records_no_retry(self):
        client = Mock()
        client.query.return_value = TestToolCallingBenchmarkCandidatePath._websocket_result(
            "websearch"
        )
        client_factory = Mock()

        observations = benchmark.collect_candidate_responses(
            {"name": "model", "provider": "provider"},
            [self._scenario("scenario-1", "Question")],
            client,
            client_factory=client_factory,
        )

        client_factory.assert_not_called()
        policy = observations["scenario-1"].stream_summary["execution_policy"]
        self.assertEqual(policy["mode"], "sequential")
        self.assertEqual(policy["concurrency"], 1)
        self.assertEqual(policy["retry_count"], 0)
        self.assertFalse(policy["retry_enabled"])
        self.assertEqual(observations["scenario-1"].stream_summary["attempt_count"], 1)

    def test_concurrency_requires_a_client_factory(self):
        with self.assertRaisesRegex(ValueError, "client_factory"):
            benchmark.collect_candidate_responses(
                {"name": "model", "provider": "provider"},
                [self._scenario("scenario-1", "Question")],
                Mock(),
                concurrency=2,
            )

    def test_bounded_concurrency_uses_independent_clients_and_preserves_order(self):
        import time

        created_clients = []
        completed_questions = []

        class ScenarioClient:
            def __init__(self, client_id: int):
                self.client_id = client_id

            def query(self, request):
                if request["query"] == "First question":
                    time.sleep(0.05)
                completed_questions.append(request["query"])
                return TestToolCallingBenchmarkCandidatePath._websocket_result("websearch")

        def client_factory():
            client = ScenarioClient(len(created_clients))
            created_clients.append(client)
            return client

        scenarios = [
            self._scenario("scenario-first", "First question"),
            self._scenario("scenario-second", "Second question"),
        ]
        observations = benchmark.collect_candidate_responses(
            {"name": "model", "provider": "provider"},
            scenarios,
            Mock(),
            concurrency=2,
            client_factory=client_factory,
        )

        self.assertEqual(list(observations), ["scenario-first", "scenario-second"])
        self.assertEqual(len(created_clients), 2)
        self.assertEqual(len({id(client) for client in created_clients}), 2)
        self.assertEqual(completed_questions, ["Second question", "First question"])
        policy = observations["scenario-first"].stream_summary["execution_policy"]
        self.assertEqual(policy["mode"], "bounded_thread_pool")
        self.assertEqual(policy["concurrency"], 2)
        self.assertFalse(policy["socket_sharing"])
        self.assertTrue(policy["independent_client_per_scenario"])

    def test_retries_are_opt_in_bounded_and_recorded_per_attempt(self):
        client = Mock()
        client.query.side_effect = [
            self._transport_result(),
            TestToolCallingBenchmarkCandidatePath._websocket_result("websearch"),
        ]

        observations = benchmark.collect_candidate_responses(
            {"name": "model", "provider": "provider"},
            [self._scenario("scenario-1", "Question")],
            client,
            retry_count=1,
        )

        observation = observations["scenario-1"]
        self.assertEqual(client.query.call_count, 2)
        self.assertEqual(observation.status, "completed")
        self.assertEqual(
            observation.stream_summary["attempt_statuses"],
            ["transport_error", "completed"],
        )
        self.assertEqual(observation.stream_summary["attempt_count"], 2)
        policy = observation.stream_summary["execution_policy"]
        self.assertEqual(policy["retry_count"], 1)
        self.assertTrue(policy["retry_enabled"])

    def test_execution_policy_bounds_are_enforced(self):
        with self.assertRaises(ValueError):
            benchmark.build_execution_policy(concurrency=0)
        with self.assertRaises(ValueError):
            benchmark.build_execution_policy(concurrency=benchmark.MAX_CONCURRENCY + 1)
        with self.assertRaises(ValueError):
            benchmark.build_execution_policy(retry_count=benchmark.MAX_RETRY_COUNT + 1)


if __name__ == "__main__":
    unittest.main()
