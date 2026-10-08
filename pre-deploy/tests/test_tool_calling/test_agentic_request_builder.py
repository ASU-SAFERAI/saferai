import unittest

from custom_benchmarks.tool_calling.agentic.runner import (
    SUPPORTED_TOOLS,
    build_agentic_request,
    build_multi_tool_payload,
    build_query_payload,
)
from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY


class TestAgenticRequestBuilder(unittest.TestCase):
    def test_builds_contract_from_enabled_tool_map_in_registry_order(self):
        enabled_tools = {
            "websearch": True,
            "create_artifact": True,
            "isearch_search": True,
            "generate_image": False,
        }

        request = build_agentic_request(
            "Find the relevant information.",
            enabled_tools,
            model_name="candidate-model",
            model_provider="candidate-provider",
        )

        self.assertEqual(request["action"], "queryV2")
        self.assertTrue(request["agentic"])
        self.assertEqual(request["query"], "Find the relevant information.")
        self.assertEqual(request["model_name"], "candidate-model")
        self.assertEqual(request["model_provider"], "candidate-provider")
        self.assertEqual(
            request["agent_config"],
            {
                "tools": [
                    tool_name
                    for tool_name in TOOL_REGISTRY
                    if enabled_tools.get(tool_name) is True
                ]
            },
        )
        self.assertEqual(request["response_format"], {"type": "json"})
        self.assertNotIn("system_prompt", request)
        self.assertNotIn("system_prompt", request["agent_config"])

    def test_iterable_tools_are_deduplicated_and_ordered_by_registry(self):
        request = build_agentic_request(
            "Use the enabled tools.",
            ("create_artifact", "websearch", "create_artifact"),
        )

        self.assertEqual(
            request["agent_config"]["tools"],
            ["websearch", "create_artifact"],
        )

    def test_compatibility_helpers_use_generalized_contract(self):
        single = build_query_payload(
            "websearch",
            "Search this.",
            model_name="candidate",
            model_provider="provider",
        )
        multi = build_multi_tool_payload(
            ("create_artifact", "isearch_search"),
            "Complete this.",
        )

        self.assertEqual(single["agent_config"]["tools"], ["websearch"])
        self.assertEqual(
            multi["agent_config"]["tools"],
            ["isearch_search", "create_artifact"],
        )
        self.assertEqual(single["model_name"], "candidate")
        self.assertEqual(single["model_provider"], "provider")
        self.assertEqual(set(SUPPORTED_TOOLS), set(TOOL_REGISTRY))

    def test_rejects_unknown_tools_and_empty_question(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool"):
            build_agentic_request("Question", ("not_registered",))

        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            build_agentic_request("  ", ("websearch",))

    def test_does_not_accept_a_string_as_the_tool_collection(self):
        with self.assertRaisesRegex(TypeError, "mapping or iterable"):
            build_agentic_request("Question", "websearch")


if __name__ == "__main__":
    unittest.main()
