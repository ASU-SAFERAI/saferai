import json
import json
import unittest

from custom_benchmarks.tool_calling.agentic.client import WebSocketQueryResult
from custom_benchmarks.tool_calling.agentic.observation import (
    build_tool_call_observation,
    classify_trace_status,
    extract_canonical_calls,
)
from custom_benchmarks.tool_calling.agentic.protocol import StreamTrace


class TestAgenticObservationExtraction(unittest.TestCase):
    @staticmethod
    def _frame(tool_call):
        return json.dumps({"tool_call": tool_call})

    def test_merges_fragments_preserves_progress_result_and_provider_payloads(self):
        trace = StreamTrace()
        trace.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-1",
            "function": {"arguments": '{"query":"find '},
            "metadata": {"provider_event": "start"},
        }))
        trace.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-1",
            "function": {"arguments": 'research"}'},
        }))
        trace.add(self._frame({
            "type": "tool_progress",
            "name": "websearch",
            "tool_call_id": "call-1",
            "content": {"status": "running"},
        }))
        trace.add(self._frame({
            "type": "tool_result",
            "name": "websearch",
            "tool_call_id": "call-1",
            "result": {"items": ["one"]},
        }))
        trace.add(json.dumps({"response": "<EOS>"}))

        calls = extract_canonical_calls(trace)

        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call.ordinal, 0)
        self.assertEqual(call.tool_name, "websearch")
        self.assertEqual(call.tool_call_id, "call-1")
        self.assertEqual(call.arguments, {"query": "find research"})
        self.assertEqual(call.argument_text, '{"query":"find research"}')
        self.assertEqual(call.tool_output, {"items": ["one"]})
        self.assertEqual(call.event_type, "mixed")
        self.assertEqual(call.metadata["event_types"], {
            "function": 2,
            "tool_progress": 1,
            "tool_result": 1,
        })
        self.assertEqual(call.metadata["stream_output"]["progress"], [{"status": "running"}])
        self.assertEqual(call.raw["raw_events"][0]["payload"]["function"]["arguments"], '{"query":"find ')

    def test_repeated_calls_remain_ordered_and_separate(self):
        trace = StreamTrace()
        for call_id, query in (("call-1", "first"), ("call-2", "second")):
            trace.add(self._frame({
                "type": "function",
                "name": "websearch",
                "tool_call_id": call_id,
                "function": {"arguments": json.dumps({"query": query})},
            }))
        trace.add(json.dumps({"response": "<EOS>"}))

        calls = extract_canonical_calls(trace, include_raw_payloads=False)

        self.assertEqual([(call.ordinal, call.tool_call_id) for call in calls], [
            (0, "call-1"),
            (1, "call-2"),
        ])
        self.assertEqual([call.arguments["query"] for call in calls], ["first", "second"])
        self.assertIsNone(calls[0].raw)

    def test_invalid_arguments_remain_visible_with_parse_error(self):
        trace = StreamTrace()
        trace.add(self._frame({
            "type": "function",
            "name": "create_artifact",
            "tool_call_id": "call-bad",
            "function": {"arguments": '{"prompt":'},
        }))
        trace.add(json.dumps({"response": "<EOS>"}))

        call = extract_canonical_calls(trace)[0]

        self.assertIsNone(call.arguments)
        self.assertEqual(call.argument_text, '{"prompt":')
        self.assertTrue(call.argument_parse_error.startswith("JSONDecodeError:"))
        self.assertEqual(call.tool_call_id, "call-bad")

    def test_progress_without_function_arguments_is_retained(self):
        trace = StreamTrace()
        trace.add(self._frame({
            "type": "tool_progress",
            "name": "websearch",
            "tool_call_id": "call-progress",
            "content": "searching",
            "render": {"kind": "status"},
        }))
        trace.add(json.dumps({"response": "<EOS>"}))

        call = extract_canonical_calls(trace)[0]

        self.assertIsNone(call.arguments)
        self.assertIsNone(call.argument_parse_error)
        self.assertEqual(call.tool_output, {
            "content": "searching",
            "renders": [{"kind": "status"}],
        })
        self.assertEqual(call.metadata["stream_output"]["content"], "searching")

    def test_classifies_each_trace_status_fixture(self):
        completed = StreamTrace()
        completed.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-complete",
            "function": {"arguments": '{"query":"find"}'},
        }))
        completed.add(json.dumps({"response": "<EOS>"}))

        abstention = StreamTrace()
        abstention.add(json.dumps({"response": "<EOS>"}))

        completed_no_call = StreamTrace()
        completed_no_call.add(json.dumps({"response": "<EOS>"}))

        malformed = StreamTrace()
        malformed.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-malformed",
            "function": {"arguments": '{"query":'},
        }))
        malformed.add(json.dumps({"response": "<EOS>"}))

        error_eos = StreamTrace()
        error_eos.add(json.dumps({"error": "provider failed", "error_type": "provider_error"}))
        error_eos.add(json.dumps({
            "error": "<EOS>",
            "error_type": "provider_error",
        }))

        transport_error = StreamTrace()
        transport_error.mark_transport_error("WebSocket closed before EOS")

        incomplete = StreamTrace()
        incomplete.add(json.dumps({"text_delta": "partial"}))

        fixtures = [
            (completed, 1, "completed", True),
            (abstention, 0, "abstention", True),
            (completed_no_call, 1, "completed_no_call", True),
            (malformed, 1, "malformed_arguments", False),
            (error_eos, 1, "error_eos", False),
            (transport_error, 1, "transport_error", False),
            (incomplete, 1, "incomplete", False),
        ]
        for trace, expected_count, expected_status, expected_completed in fixtures:
            with self.subTest(status=expected_status):
                classification = classify_trace_status(
                    trace,
                    expected_call_count=expected_count,
                )
                self.assertEqual(classification.status, expected_status)
                self.assertEqual(classification.completed, expected_completed)
                self.assertTrue(classification.status_reason)

    def test_status_precedence_is_transport_then_stream_error_then_malformed(self):
        malformed_error = StreamTrace()
        malformed_error.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-precedence",
            "function": {"arguments": "{"},
        }))
        malformed_error.add(json.dumps({"error": "<EOS>", "error_type": "provider_error"}))
        self.assertEqual(
            classify_trace_status(malformed_error, expected_call_count=1).status,
            "error_eos",
        )

        transport_error = StreamTrace()
        transport_error.add(self._frame({
            "type": "function",
            "name": "websearch",
            "function": {"arguments": "{"},
        }))
        transport_error.add(json.dumps({"error": "<EOS>", "error_type": "provider_error"}))
        transport_error.mark_transport_error("connection closed")
        self.assertEqual(
            classify_trace_status(transport_error, expected_call_count=1).status,
            "transport_error",
        )

    def test_classifier_can_reuse_extracted_calls(self):
        trace = StreamTrace()
        trace.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-reused",
            "function": {"arguments": '{"query":"find"}'},
        }))
        trace.add(json.dumps({"response": "<EOS>"}))
        calls = extract_canonical_calls(trace, include_raw_payloads=False)

        classification = classify_trace_status(
            trace,
            expected_call_count=1,
            calls=calls,
        )

        self.assertEqual(classification.status, "completed")
        self.assertTrue(classification.completed)

    def test_build_observation_preserves_trace_metadata_and_redacts_request(self):
        trace = StreamTrace()
        trace.add(json.dumps({"text_delta": "final answer"}))
        trace.add(json.dumps({"thinking": "private reasoning"}))
        trace.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-observed",
            "function": {"arguments": '{"query":"find"}'},
        }))
        trace.add(self._frame({
            "type": "tool_result",
            "name": "websearch",
            "tool_call_id": "call-observed",
            "result": {"items": ["one"]},
        }))
        trace.add(json.dumps({
            "response": "<EOS>",
            "metadata": {
                "query_id": "query-1",
                "usage_metric": {"total_token_count": 12},
            },
        }))

        result = WebSocketQueryResult(
            request={
                "query": "find",
                "authorization": "Bearer request-secret",
                "agent_config": {"api_key": "request-secret"},
            },
            trace=trace,
        )
        observation = build_tool_call_observation(
            result,
            run_id="run-1",
            scenario_id="scenario-1",
            candidate={"name": "model", "provider": "provider"},
            expected_call_count=1,
        )

        self.assertEqual(observation.status, "completed")
        self.assertTrue(observation.completed)
        self.assertTrue(observation.eos_received)
        self.assertEqual(observation.final_response, "final answer")
        self.assertEqual(observation.thinking, "private reasoning")
        self.assertEqual(observation.tool_calls[0].tool_output, {"items": ["one"]})
        self.assertEqual(observation.eos_metadata["query_id"], "query-1")
        self.assertEqual(observation.stream_summary["frame_count"], 5)
        self.assertEqual(observation.stream_summary["tool_call_count"], 2)
        self.assertEqual(observation.request["authorization"], "<redacted>")
        self.assertEqual(observation.request["agent_config"]["api_key"], "<redacted>")
        self.assertIsNone(observation.raw_trace)
        self.assertIsNone(observation.tool_calls[0].raw)

    def test_build_observation_retains_redacted_raw_frames_when_opted_in(self):
        trace = StreamTrace()
        trace.add(self._frame({
            "type": "function",
            "name": "websearch",
            "tool_call_id": "call-raw",
            "metadata": {"access_token": "frame-secret"},
            "function": {"arguments": '{"query":"find"}'},
        }))
        trace.add(json.dumps({
            "response": "<EOS>",
            "metadata": {"authorization": "Bearer eos-secret"},
        }))
        result = WebSocketQueryResult(request={"query": "find"}, trace=trace)

        observation = build_tool_call_observation(
            result,
            run_id="run-raw",
            scenario_id="scenario-raw",
            candidate={"name": "model", "provider": "provider"},
            expected_call_count=1,
            include_raw_frames=True,
        )

        self.assertIsNotNone(observation.raw_trace)
        self.assertIn("frames", observation.raw_trace)
        self.assertEqual(
            observation.raw_trace["eos_metadata"]["authorization"],
            "<redacted>",
        )
        self.assertEqual(observation.tool_calls[0].raw["metadata"]["access_token"], "<redacted>")
        self.assertEqual(
            observation.tool_calls[0].metadata["stream_output"]["function_arguments"],
            '{"query":"find"}',
        )

    def test_build_observation_preserves_error_and_transport_statuses(self):
        error_trace = StreamTrace()
        error_trace.add(json.dumps({"error": "provider failed", "error_type": "provider_error"}))
        error_trace.add(json.dumps({
            "error": "<EOS>",
            "error_type": "provider_error",
            "metadata": {"token": "eos-secret"},
        }))
        error_observation = build_tool_call_observation(
            WebSocketQueryResult(request={}, trace=error_trace),
            run_id="run-error",
            scenario_id="scenario-error",
            candidate={"name": "model", "provider": "provider"},
            expected_call_count=1,
        )
        self.assertEqual(error_observation.status, "error_eos")
        self.assertFalse(error_observation.completed)
        self.assertEqual(error_observation.errors[0]["error_type"], "provider_error")
        self.assertEqual(error_observation.eos_metadata["token"], "<redacted>")

        transport_trace = StreamTrace()
        transport_trace.add(json.dumps({"text_delta": "partial"}))
        transport_trace.mark_transport_error("Bearer transport-secret")
        transport_observation = build_tool_call_observation(
            WebSocketQueryResult(request={}, trace=transport_trace),
            run_id="run-transport",
            scenario_id="scenario-transport",
            candidate={"name": "model", "provider": "provider"},
            expected_call_count=1,
        )
        self.assertEqual(transport_observation.status, "transport_error")
        self.assertEqual(transport_observation.transport_error, "Bearer <redacted>")
        self.assertEqual(transport_observation.final_response, "partial")


if __name__ == "__main__":
    unittest.main()
