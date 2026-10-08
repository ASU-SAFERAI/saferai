"""Combination generation for tool-calling evaluation.

Provides a standalone function to generate all non-empty subsets
of the tools in the registry.
"""

import itertools

from ..registry import TOOL_REGISTRY

__all__ = ["generate_combinations"]


def generate_combinations() -> list[dict]:
    """Generate all non-empty subsets of the registry's tools.

    Produces every non-empty subset of the tools defined in TOOL_REGISTRY
    (2**n - 1 combinations for n tools; e.g. 31 for 5 tools). Each entry has
    a boolean field per registry tool and a unique sequential identifier. IDs
    are assigned deterministically, ordered by subset size then registry order.

    Returns
    -------
    list[dict]
        List of combination dictionaries with combination_id,
        boolean tool fields, and tools_enabled list.
    """
    tool_keys = list(TOOL_REGISTRY.keys())
    combinations_list = []
    combo_index = 1

    # Generate subsets of size 1 through n (ordered by size, then registry order)
    for size in range(1, len(tool_keys) + 1):
        for subset in itertools.combinations(tool_keys, size):
            enabled_set = set(subset)
            combination = {
                "combination_id": f"combo_{combo_index:02d}",
                **{key: key in enabled_set for key in tool_keys},
                "tools_enabled": list(subset),
            }
            combinations_list.append(combination)
            combo_index += 1

    return combinations_list
