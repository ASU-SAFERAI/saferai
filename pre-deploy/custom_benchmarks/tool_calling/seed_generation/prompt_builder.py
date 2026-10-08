"""System prompt builder for tool-calling evaluation.

Provides a standalone function to dynamically construct system prompts
based on a set of enabled tools from the registry.
"""

import json

from ..registry import TOOL_REGISTRY

__all__ = ["build_system_prompt"]


def build_system_prompt(enabled_tools: list[str]) -> str:
    """Dynamically construct a system prompt based on enabled tools.

    Assembles documentation blocks for each enabled tool from the
    TOOL_REGISTRY, ordered by registry definition order.

    Parameters
    ----------
    enabled_tools : list[str]
        List of tool identifiers to include in the system prompt.

    Returns
    -------
    str
        The assembled system prompt string.
    """
    base_prompt = (
        "You are a helpful assistant with access to the following tools. "
        "Select the single most appropriate tool for the user's request.\n\n"
        "Output rules (follow exactly):\n"
        "1. Respond with ONLY a single JSON object and nothing else. Do not "
        "include prose, explanations, markdown fences, or any text before or "
        "after the JSON.\n"
        "2. Emit exactly ONE tool call. If the request requires a sequence of "
        "tool calls, output ONLY the first tool call needed to begin the "
        "sequence. Never emit multiple tool calls in a single response.\n"
        "3. The JSON object must have this shape: "
        '{"tool_name": "<tool>", "arguments": {<parameters>}}.\n'
        "4. If no available tool is appropriate for the request, respond with "
        'exactly {"tool_name": null, "arguments": {}}.'
    )

    if not enabled_tools:
        return base_prompt

    # Filter to only valid tool identifiers and maintain registry order
    tool_blocks = []
    for tool_key in TOOL_REGISTRY:
        if tool_key not in enabled_tools:
            continue
        tool = TOOL_REGISTRY[tool_key]
        block_lines = [
            f"## {tool['name']}",
            f"Description: {tool['description']}",
        ]

        # Required parameters
        if tool["required_parameters"]:
            block_lines.append("Required Parameters:")
            for param in tool["required_parameters"]:
                block_lines.append(
                    f"  - {param['name']} ({param['type']}): {param['constraints']}"
                )

        # Optional parameters
        if tool["optional_parameters"]:
            block_lines.append("Optional Parameters:")
            for param in tool["optional_parameters"]:
                default_str = (
                    json.dumps(param["default"])
                    if param["default"] is not None
                    else "null"
                )
                block_lines.append(
                    f"  - {param['name']} ({param['type']}, default: {default_str}): "
                    f"{param['constraints']}"
                )

        # Usage instructions
        block_lines.append(
            f"Usage: Call this tool by specifying the tool name '{tool_key}' "
            f"and providing the required parameters."
        )

        tool_blocks.append("\n".join(block_lines))

    if not tool_blocks:
        return base_prompt

    reminder = (
        "Remember: output ONLY one JSON tool call (or the null-tool JSON if no "
        "tool fits), with no surrounding text. For multi-step requests, emit "
        "only the first tool call in the sequence."
    )

    return (
        base_prompt
        + "\n\n"
        + "\n\n".join(tool_blocks)
        + "\n\n"
        + reminder
    )
