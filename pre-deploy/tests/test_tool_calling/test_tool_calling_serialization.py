import json
import unittest
from datetime import datetime
from decimal import Decimal

from custom_benchmarks.tool_calling.models import CanonicalCall, GoldenCall
from custom_benchmarks.tool_calling.serialization import (
    canonical_json,
    serialize_expected_trace,
    serialize_observed_trace,
)


class TestToolCallingSerialization(unittest.TestCase):
    def test_expected_trace_preserves_order_and_symbolic_placeholders(self):
        source = [
            GoldenCall(
                tool_name="websearch",
                arguments={"query": "find this", "options": {"limit": 3}},
            ),
            GoldenCall(
                tool_name="create_artifact",
                arguments={"prompt": "Use {{steps[0].output}}"},
            ),
        ]

        rendered = serialize_expected_trace(source)

        self.assertEqual(
            rendered,
            '{"tool_calls":[{"arguments":{"options":{"limit":3},"query":"find this"},"tool_name":"websearch"},{"arguments":{"prompt":"Use {{steps[0].output}}"},"tool_name":"create_artifact"}]}',
        )
        self.assertEqual(json.loads(rendered)["tool_calls"][1]["arguments"]["prompt"], "Use {{steps[0].output}}")

    def test_explicit_null_abstention_serializes_as_empty_trace(self):
        rendered = serialize_expected_trace([
            GoldenCall(tool_name=None, arguments={}),
        ])

        self.assertEqual(json.loads(rendered), {"tool_calls": []})
        self.assertEqual(serialize_expected_trace([]), rendered)

    def test_observed_trace_keeps_malformed_arguments_outputs_and_call_metadata(self):
        call = CanonicalCall(
            ordinal=4,
            tool_name="websearch",
            arguments=None,
            argument_text='{"query":',
            argument_parse_error="JSONDecodeError",
            tool_call_id="call-4",
            event_type="function",
            tool_output={"items": [Decimal("1.20"), b"ok"]},
            metadata={"provider": {"attempt": 1}},
        )

        rendered = serialize_observed_trace(
            [call],
            metadata={"scenario": "scenario-1", "captured_at": datetime(2025, 1, 2, 3, 4, 5)},
        )
        payload = json.loads(rendered)
        observed = payload["tool_calls"][0]

        self.assertIsNone(observed["arguments"])
        self.assertEqual(observed["argument_text"], '{"query":')
        self.assertEqual(observed["argument_parse_error"], "JSONDecodeError")
        self.assertEqual(observed["tool_call_id"], "call-4")
        self.assertEqual(observed["tool_output"]["items"][0], {"__type__": "decimal", "value": "1.20"})
        self.assertEqual(observed["tool_output"]["items"][1]["__type__"], "bytes")
        self.assertEqual(observed["metadata"]["provider"]["attempt"], 1)
        self.assertEqual(payload["metadata"]["scenario"], "scenario-1")
        self.assertNotIn("final_response", payload)
        self.assertNotIn("thinking", payload)

    def test_canonical_json_is_stable_for_nested_non_native_values(self):
        first = {"z": {3, 1, 2}, "a": {"b": Decimal("2.0"), "a": "é"}}
        second = {"a": {"a": "é", "b": Decimal("2.0")}, "z": {2, 3, 1}}

        self.assertEqual(canonical_json(first), canonical_json(second))
        self.assertEqual(
            canonical_json({"value": float("inf")}),
            '{"value":{"__type__":"float","value":"inf"}}',
        )

    def test_mapping_observation_uses_parsed_arguments_and_never_final_prose(self):
        rendered = serialize_observed_trace(
            {
                "tool_calls": [
                    {
                        "tool_name": "isearch_search",
                        "arguments": {"query": "hello"},
                        "tool_output": {"answer": "result"},
                    }
                ],
                "final_response": "This prose must not be scored",
            }
        )

        self.assertEqual(
            json.loads(rendered),
            {
                "tool_calls": [
                    {
                        "arguments": {"query": "hello"},
                        "ordinal": 0,
                        "tool_name": "isearch_search",
                        "tool_output": {"answer": "result"},
                    }
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
