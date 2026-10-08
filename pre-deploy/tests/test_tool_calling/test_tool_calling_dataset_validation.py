import copy
import json
import unittest
from pathlib import Path

from custom_benchmarks.tool_calling.dataset_validation import (
    DatasetValidationError,
    validate_dataset_file,
    validate_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    PROJECT_ROOT
    / "custom_benchmarks"
    / "tool_calling"
    / "tool_calling_seed_60_populated_multihop.json"
)


class TestToolCallingDatasetValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    def test_canonical_multihop_dataset_has_complete_ordered_truth(self):
        validated = validate_dataset_file(DATASET_PATH)

        self.assertEqual(len(validated), 60)
        self.assertEqual(sum(scenario["hop_count"] == 0 for scenario in validated), 18)
        self.assertGreater(
            sum("{{steps[" in json.dumps(scenario["golden_truth"]) for scenario in validated),
            0,
        )
        for scenario in validated:
            if scenario["hop_count"] == 0:
                self.assertEqual(scenario["expected_tools"], [])
                self.assertEqual(
                    scenario["golden_truth"],
                    [{"tool_name": None, "arguments": {}}],
                )
            else:
                self.assertEqual(len(scenario["golden_truth"]), scenario["hop_count"])
                self.assertEqual(
                    [call["tool_name"] for call in scenario["golden_truth"]],
                    scenario["expected_tools"],
                )

    def test_only_canonical_migrated_dataset_is_accepted_populated_source(self):
        """The legacy populated seed must not become a multi-hop fallback."""
        dataset_dir = DATASET_PATH.parent
        candidate_paths = sorted(dataset_dir.glob("tool_calling_seed_60_populated*.json"))
        self.assertIn(DATASET_PATH, candidate_paths)

        accepted_paths = []
        rejected_paths = {}
        for path in candidate_paths:
            try:
                validate_dataset_file(path)
            except DatasetValidationError as exc:
                rejected_paths[path.name] = str(exc)
            else:
                accepted_paths.append(path)

        self.assertEqual(accepted_paths, [DATASET_PATH])
        self.assertIn("tool_calling_seed_60_populated.json", rejected_paths)
        self.assertIn("golden_truth", rejected_paths["tool_calling_seed_60_populated.json"])

    def test_rejects_golden_truth_length_and_order_mismatch(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 2))
        scenario["golden_truth"] = scenario["golden_truth"][:1]
        scenario["expected_tools"] = list(reversed(scenario["expected_tools"]))

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        message = str(context.exception)
        self.assertIn("length", message)
        self.assertIn("must equal expected_tools[0]", message)

    def test_rejects_missing_or_invalid_registry_arguments(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["expected_tools"] == ["websearch"]))
        arguments = scenario["golden_truth"][0]["arguments"]
        arguments.pop("query")
        arguments["limit"] = 0

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        message = str(context.exception)
        self.assertIn("missing required parameter 'query'", message)
        self.assertIn("must be between 1 and 100", message)

    def test_rejects_current_or_future_placeholder_references(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 2))
        scenario["golden_truth"][1]["arguments"]["prompt"] = "Use {{steps[1].output}}"

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        self.assertIn("only steps before hop 1 are valid", str(context.exception))

    def test_accepts_empty_and_explicit_null_abstention_truth(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 0))

        scenario["golden_truth"] = []
        validate_scenarios([scenario])

        scenario["golden_truth"] = [{"tool_name": None, "arguments": {}}]
        validate_scenarios([scenario])

    def test_rejects_missing_blank_and_duplicate_scenario_ids_and_questions(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 0))
        scenario["scenario_id"] = "duplicate-id"
        scenario["questions"] = ["A valid question"]

        duplicate = copy.deepcopy(scenario)
        missing_id = copy.deepcopy(scenario)
        missing_id.pop("scenario_id")
        blank_question = copy.deepcopy(scenario)
        blank_question["scenario_id"] = "blank-question"
        blank_question["questions"] = ["  "]

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario, duplicate, missing_id, blank_question])

        message = str(context.exception)
        self.assertIn("scenario_id: must be unique", message)
        self.assertIn("scenario_id: must be a non-empty string", message)
        self.assertIn("questions[0]: must be a non-empty string", message)

    def test_rejects_unknown_or_disabled_expected_tools_and_enabled_registry_keys(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 1))
        scenario["enabled_tools"]["not_a_registry_tool"] = True
        scenario["enabled_tools"]["websearch"] = False
        scenario["expected_tools"] = ["websearch"]

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        message = str(context.exception)
        self.assertIn("unknown registry tool 'not_a_registry_tool'", message)
        self.assertIn("expected_tools[0]", message)
        self.assertIn("tool must be enabled for the scenario", message)

    def test_rejects_null_for_required_argument(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["expected_tools"] == ["websearch"]))
        scenario["golden_truth"][0]["arguments"]["query"] = None

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        self.assertIn("arguments['query']: must be a string", str(context.exception))

    def test_rejects_malformed_placeholder_dependencies(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 2))
        scenario["golden_truth"][1]["arguments"]["prompt"] = "Use {{steps[0].result}}"

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        self.assertIn("contains a malformed step output placeholder", str(context.exception))

    def test_rejects_non_null_call_in_abstention_truth(self):
        scenario = copy.deepcopy(next(item for item in self.scenarios if item["hop_count"] == 0))
        scenario["golden_truth"] = [{"tool_name": "websearch", "arguments": {"query": "unexpected"}}]

        with self.assertRaises(DatasetValidationError) as context:
            validate_scenarios([scenario])

        self.assertIn("must be empty or the explicit null-tool abstention object", str(context.exception))


if __name__ == "__main__":
    unittest.main()
