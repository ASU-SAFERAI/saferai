import unittest

from custom_benchmarks.tool_calling.catalog import project_tool_catalog
from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY


class TestToolCallingCatalog(unittest.TestCase):
    def test_projects_enabled_tools_in_registry_order_with_full_schema(self):
        catalog = project_tool_catalog(
            {
                "create_artifact": True,
                "isearch_search": False,
                "websearch": True,
            }
        )

        self.assertEqual(list(catalog), ["websearch", "create_artifact"])
        self.assertEqual(
            list(catalog["websearch"]),
            [
                "name",
                "description",
                "required_parameters",
                "optional_parameters",
                "endpoint",
                "return_schema",
            ],
        )
        self.assertEqual(catalog["websearch"]["name"], "Web Search")
        self.assertEqual(
            catalog["websearch"]["required_parameters"][0]["name"],
            "query",
        )
        self.assertEqual(catalog["websearch"]["optional_parameters"][0]["default"], 5)
        self.assertIn("Range 1-100", catalog["websearch"]["optional_parameters"][0]["constraints"])
        self.assertIsNone(catalog["websearch"]["endpoint"])
        self.assertEqual(catalog["websearch"]["return_schema"], [{"name": "results", "type": "array"}])

    def test_accepts_enabled_name_iterables_and_returns_independent_definitions(self):
        catalog = project_tool_catalog(["generate_image", "isearch_search"])

        self.assertEqual(list(catalog), ["isearch_search", "generate_image"])
        catalog["generate_image"]["required_parameters"][0]["name"] = "changed"
        self.assertEqual(
            TOOL_REGISTRY["generate_image"]["required_parameters"][0]["name"],
            "prompt",
        )

    def test_rejects_unknown_tools_even_when_disabled(self):
        with self.assertRaisesRegex(ValueError, "not_registered"):
            project_tool_catalog({"not_registered": False})

        with self.assertRaisesRegex(ValueError, "not_registered"):
            project_tool_catalog(["not_registered"])

    def test_rejects_non_boolean_mapping_flags_and_non_string_names(self):
        with self.assertRaisesRegex(TypeError, "booleans"):
            project_tool_catalog({"websearch": 1})

        with self.assertRaisesRegex(TypeError, "strings"):
            project_tool_catalog(["websearch", 3])


if __name__ == "__main__":
    unittest.main()
