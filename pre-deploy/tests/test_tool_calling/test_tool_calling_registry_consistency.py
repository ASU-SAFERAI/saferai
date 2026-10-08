import unittest

from custom_benchmarks.tool_calling.agentic.runner import (
    DEFAULT_MULTI_TOOL_QUERY,
    DEFAULT_TOOL_QUERIES,
    SUPPORTED_TOOLS,
    build_agentic_request,
)
from custom_benchmarks.tool_calling.catalog import project_tool_catalog
from custom_benchmarks.tool_calling.seed_generation import generate_combinations
from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY


class TestToolCallingRegistryConsistency(unittest.TestCase):
    def test_request_catalog_and_probe_defaults_share_registry_order(self):
        enabled_tools = {tool_name: True for tool_name in TOOL_REGISTRY}
        request = build_agentic_request("Use the registered tools.", enabled_tools)
        catalog = project_tool_catalog(enabled_tools)

        self.assertEqual(tuple(SUPPORTED_TOOLS), tuple(TOOL_REGISTRY))
        self.assertEqual(request["agent_config"]["tools"], list(TOOL_REGISTRY))
        self.assertEqual(list(catalog), list(TOOL_REGISTRY))
        self.assertEqual(tuple(DEFAULT_TOOL_QUERIES), tuple(TOOL_REGISTRY))
        self.assertTrue(all(query.strip() for query in DEFAULT_TOOL_QUERIES.values()))
        self.assertTrue(
            all(tool_name in DEFAULT_MULTI_TOOL_QUERY for tool_name in TOOL_REGISTRY)
        )

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


if __name__ == "__main__":
    unittest.main()
