import unittest

from pre_deploy.dataset_generation.tool_calling import (
    TOOL_REGISTRY,
    ToolCallingSeedGenerator,
    generate_combinations,
    generate_scenarios,
)


class TestToolCallingDatasetGeneration(unittest.TestCase):
    def test_combinations_follow_current_registry_cardinality_and_shape(self):
        combinations = generate_combinations()

        self.assertEqual(len(combinations), 2 ** len(TOOL_REGISTRY) - 1)
        for combination in combinations:
            tool_fields = {
                key
                for key in combination
                if key not in {"combination_id", "tools_enabled"}
            }
            self.assertEqual(tool_fields, set(TOOL_REGISTRY))
            self.assertEqual(
                combination["tools_enabled"],
                [tool_name for tool_name in TOOL_REGISTRY if combination[tool_name]],
            )

    def test_generator_produces_structure_only_scenarios(self):
        generator = ToolCallingSeedGenerator(num_variants=1)
        scenarios = generator.generate_scenarios()

        self.assertTrue(scenarios)
        for scenario in scenarios:
            # Structure-only: question text and golden arguments are filled in
            # by a later step, so the generator emits empty placeholders.
            self.assertIn("scenario_id", scenario)
            self.assertIn("combination_id", scenario)
            self.assertIn("hop_count", scenario)
            self.assertEqual(scenario.get("questions"), [])
            for call in scenario.get("golden_truth", []):
                self.assertEqual(call.get("arguments"), {})

    def test_generate_scenarios_is_deterministic_for_fixed_variants(self):
        self.assertEqual(generate_scenarios(1), generate_scenarios(1))

    def test_generator_rejects_invalid_variant_counts(self):
        with self.assertRaises(ValueError):
            ToolCallingSeedGenerator(num_variants=0)
        with self.assertRaises(ValueError):
            ToolCallingSeedGenerator(num_variants=True)


if __name__ == "__main__":
    unittest.main()
