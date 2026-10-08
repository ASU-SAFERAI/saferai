"""Tool registry for tool-calling discrimination evaluation.

Contains the canonical TOOL_REGISTRY constant defining the live tool set
with its documentation, parameters, and return schemas.
"""

__all__ = ["TOOL_REGISTRY"]

# Module-level constant: structured registry of the live tools with their documentation.
# Insertion order defines the canonical tool ordering used throughout the module.
TOOL_REGISTRY: dict = {
    "isearch_search": {
        "name": "iSearch Search",
        "description": (
            "Search ASU's faculty and staff directory to find people by name, "
            "department, or keyword. Returns matching profile results including "
            "contact information."
        ),
        "required_parameters": [
            {
                "name": "query",
                "type": "string",
                "constraints": "Must match pattern [A-Za-z0-9 ]+ (letters, numbers, and spaces only)",
            }
        ],
        "optional_parameters": [],
        "endpoint": "https://search.asu.edu/api/v1/webdir-profiles/faculty-staff/filtered",
        "return_schema": [
            {
                "name": "results",
                "type": "array",
                "items": [
                    {"name": "id", "type": "string"},
                    {"name": "name", "type": "string"},
                    {"name": "email", "type": "string"},
                    {"name": "phone", "type": "string"},
                ],
            },
        ],
    },
    "generate_image": {
        "name": "Image Generation",
        "description": (
            "Generate an image based on a text prompt. Optionally accepts "
            "existing image file IDs as reference inputs for guided generation."
        ),
        "required_parameters": [
            {
                "name": "prompt",
                "type": "string",
                "constraints": "Non-empty text description of the desired image",
            }
        ],
        "optional_parameters": [
            {
                "name": "image_file_ids",
                "type": "array of strings",
                "default": None,
                "constraints": "Array of existing file ID strings to use as reference images, or null",
            }
        ],
        "endpoint": None,
        "return_schema": [
            {"name": "file_id", "type": "string"},
            {"name": "mime_type", "type": "string"},
            {"name": "filename", "type": "string"},
            {"name": "success", "type": "boolean"},
        ],
    },
    "visualize_widget": {
        "name": "Interactive Visualization",
        "description": (
            "Create an interactive visualization widget from a structured prompt. "
            "The prompt must include INTENT, COMPONENTS, CONSTRAINTS, and CONTEXT "
            "sections to fully specify the desired widget."
        ),
        "required_parameters": [
            {
                "name": "prompt",
                "type": "string",
                "constraints": (
                    "Must include INTENT, COMPONENTS, CONSTRAINTS, and CONTEXT sections"
                ),
            }
        ],
        "optional_parameters": [],
        "endpoint": None,
        "return_schema": [
            {"name": "title", "type": "string"},
            {"name": "widget_code", "type": "string"},
            {"name": "success", "type": "boolean"},
        ],
    },
    "websearch": {
        "name": "Web Search",
        "description": (
            "Perform a web search query and return results in the specified format. "
            "Supports configurable result limits, output formats, locale, and "
            "content filtering options."
        ),
        "required_parameters": [
            {
                "name": "query",
                "type": "string",
                "constraints": "Non-empty search query string",
            }
        ],
        "optional_parameters": [
            {
                "name": "limit",
                "type": "integer",
                "default": 5,
                "constraints": "Range 1-100 inclusive",
            },
            {
                "name": "formats",
                "type": "array of strings",
                "default": ["markdown"],
                "constraints": "List of desired output format strings",
            },
            {
                "name": "country",
                "type": "string",
                "default": "us",
                "constraints": "ISO country code string",
            },
            {
                "name": "lang",
                "type": "string",
                "default": "en",
                "constraints": "ISO language code string",
            },
            {
                "name": "timeout",
                "type": "integer",
                "default": 30000,
                "constraints": "Timeout in milliseconds, range 1000-300000",
            },
            {
                "name": "only_main_content",
                "type": "boolean",
                "default": True,
                "constraints": "When true, return only main page content",
            },
        ],
        "endpoint": None,
        "return_schema": [
            {"name": "results", "type": "array"},
        ],
    },
    "create_artifact": {
        "name": "Create Artifact",
        "description": (
            "Build a self-contained webpage or component from a text prompt. "
            "All HTML, CSS, and JS are embedded in a single file."
        ),
        "required_parameters": [
            {
                "name": "prompt",
                "type": "string",
                "constraints": (
                    "A clear description of the webpage/component to build. "
                    "All HTML/CSS/JS will be embedded in a single file."
                ),
            }
        ],
        "optional_parameters": [],
        "endpoint": None,
        "return_schema": [
            {"name": "artifact", "type": "string"},
            {"name": "success", "type": "boolean"},
        ],
    },
}
