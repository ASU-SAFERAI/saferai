"""Registry-backed tool catalog projections for benchmark inspection.

The agentic endpoint receives only tool identifiers.  This module provides the
separate, human/evaluator-facing projection of those identifiers into the
local registry's documented schemas.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any

from .registry import TOOL_REGISTRY


_CATALOG_FIELDS = (
    "name",
    "description",
    "required_parameters",
    "optional_parameters",
    "endpoint",
    "return_schema",
)


def _selected_tool_names(
    enabled_tools: Mapping[str, bool] | Iterable[str],
    tool_registry: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    """Normalize enabled tool mappings/lists and reject unknown identifiers."""
    if isinstance(enabled_tools, Mapping):
        unknown = [name for name in enabled_tools if name not in tool_registry]
        if unknown:
            raise ValueError(
                "Unknown tool(s) in enabled_tools: "
                + ", ".join(repr(name) for name in sorted(unknown, key=str))
            )
        invalid_flags = [
            name
            for name, enabled in enabled_tools.items()
            if not isinstance(enabled, bool)
        ]
        if invalid_flags:
            raise TypeError(
                "enabled_tools values must be booleans for: "
                + ", ".join(repr(name) for name in invalid_flags)
            )
        return {name for name, enabled in enabled_tools.items() if enabled}

    if isinstance(enabled_tools, (str, bytes)):
        raise TypeError("enabled_tools must be a mapping or iterable of tool names")

    names = list(enabled_tools)
    invalid_names = [name for name in names if not isinstance(name, str)]
    if invalid_names:
        raise TypeError("enabled tool names must be strings")
    unknown = [name for name in names if name not in tool_registry]
    if unknown:
        raise ValueError(
            "Unknown tool(s) in enabled_tools: "
            + ", ".join(repr(name) for name in sorted(set(unknown)))
        )
    return set(names)


def project_tool_catalog(
    enabled_tools: Mapping[str, bool] | Iterable[str],
    *,
    tool_registry: Mapping[str, Mapping[str, Any]] = TOOL_REGISTRY,
) -> dict[str, dict[str, Any]]:
    """Project enabled registry tools into a deterministic inspection catalog.

    The returned mapping follows ``tool_registry`` insertion order regardless
    of the input mapping/list order.  Each definition is copied so report or
    evaluator metadata cannot mutate the canonical registry.  Unknown tool
    identifiers are rejected, including identifiers marked disabled in a
    boolean mapping, so catalog generation cannot hide a dataset error.
    """
    selected_names = _selected_tool_names(enabled_tools, tool_registry)
    return {
        tool_name: {
            field: deepcopy(definition.get(field, [] if field.endswith("_parameters") or field == "return_schema" else None))
            for field in _CATALOG_FIELDS
        }
        for tool_name, definition in tool_registry.items()
        if tool_name in selected_names
    }


__all__ = ["project_tool_catalog"]
