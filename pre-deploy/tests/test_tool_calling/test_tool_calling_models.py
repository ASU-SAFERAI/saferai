import json
import unittest
from pathlib import Path

from custom_benchmarks.tool_calling.models import (
    CanonicalCall,
    GoldenCall,
    ScenarioView,
    TRACE_STATUSES,
    ToolCallObservation,
)
from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "custom_benchmarks"
    / "tool_calling"
    / "tool_calling_seed_60_populated_multihop.json"
)


class TestToolCallingModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    def test_scenario_view_adapts_legacy_seed_and_preserves_raw_record(self):
        source = next(item for item in self.scenarios if sum(item["enabled_tools"].values()) > 1)
        view = ScenarioView.from_mapping(source, tool_order=list(reversed(TOOL_REGISTRY)))

        self.assertEqual(view.scenario_id, source["scenario_id"])
        self.assertEqual(view.question, source["questions"][0])
        self.assertEqual(
            view.enabled_tools,
            [
                name
                for name in reversed(list(TOOL_REGISTRY))
                if source["enabled_tools"].get(name) is True
            ],
        )
        self.assertIsInstance(view.golden_truth[0], GoldenCall)
        self.assertEqual(view.golden_truth[0].arguments, source["golden_truth"][0]["arguments"])
        self.assertIn("system_prompt", view.raw_record)

        source["questions"][0] = "changed after model construction"
        self.assertNotEqual(view.question, source["questions"][0])

    def test_every_canonical_record_materializes_as_a_scenario_view(self):
        views = [
            ScenarioView.from_mapping(item, tool_order=list(TOOL_REGISTRY))
            for item in self.scenarios
        ]

        self.assertEqual(len(views), 60)
        self.assertTrue(all(isinstance(view, ScenarioView) for view in views))
        self.assertEqual(sum(view.hop_count == 0 for view in views), 18)

    def test_golden_call_rejects_non_object_arguments(self):
        with self.assertRaises(TypeError):
            GoldenCall.from_mapping({"tool_name": "websearch", "arguments": "not-an-object"})

    def test_canonical_call_retains_malformed_argument_for_forensics(self):
        call = CanonicalCall.from_mapping(
            {
                "ordinal": 2,
                "tool_name": "websearch",
                "arguments": None,
                "argument_text": '{"query":',
                "argument_parse_error": "JSONDecodeError",
                "tool_call_id": "call-2",
                "event_type": "function",
                "raw_provider_event": {"fragment": '{"query":'},
            }
        )

        self.assertIsNone(call.arguments)
        self.assertEqual(call.argument_text, '{"query":')
        self.assertEqual(call.argument_parse_error, "JSONDecodeError")
        self.assertEqual(call.raw["raw_provider_event"]["fragment"], '{"query":')
        self.assertEqual(call.to_dict()["ordinal"], 2)

    def test_observation_round_trip_preserves_candidate_and_trace_data(self):
        observation = ToolCallObservation(
            run_id="run-1",
            scenario_id="scenario-1",
            candidate={"name": "candidate-model", "provider": "aws"},
            request={"query": "find this", "agentic": True},
            status="malformed_arguments",
            status_reason="one function fragment was not valid JSON",
            completed=False,
            eos_received=True,
            tool_calls=[
                CanonicalCall(
                    ordinal=0,
                    tool_name="websearch",
                    arguments=None,
                    argument_text="{",
                    argument_parse_error="invalid JSON",
                    tool_output={"partial": True},
                    raw={"frame": "provider-specific"},
                )
            ],
            final_response="partial response",
            thinking="internal trace",
            errors=[{"error_type": "parse_error"}],
            transport_error=None,
            stream_summary={"frame_count": 3},
            raw_trace={"frames": [{"raw": "..."}]},
        )

        restored = ToolCallObservation.from_mapping(observation.to_dict())

        self.assertEqual(restored.model, "candidate-model")
        self.assertEqual(restored.provider, "aws")
        self.assertEqual(restored.status, "malformed_arguments")
        self.assertEqual(restored.tool_calls[0].tool_output, {"partial": True})
        self.assertEqual(restored.raw_trace, {"frames": [{"raw": "..."}]})
        self.assertEqual(set(TRACE_STATUSES), {
            "completed",
            "abstention",
            "completed_no_call",
            "malformed_arguments",
            "error_eos",
            "transport_error",
            "incomplete",
        })

    def test_observation_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            ToolCallObservation(
                run_id="run-1",
                scenario_id="scenario-1",
                candidate={"name": "model", "provider": "provider"},
                request={},
                status="unknown",  # type: ignore[arg-type]
                completed=False,
                eos_received=False,
            )


if __name__ == "__main__":
    unittest.main()
