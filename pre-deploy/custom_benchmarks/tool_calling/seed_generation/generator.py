"""Seed module for tool-calling discrimination evaluation metadata generation.

This module provides the ToolCallingSeedGenerator class and TOOL_REGISTRY constant
for generating structured evaluation metadata for tool-calling discrimination testing.
It delegates to focused submodules for registry, prompt building, combination
generation, scenario generation, and output serialization.

Primary interface is programmatic (import and instantiate). CLI execution is supported
via `python custom_benchmarks/generate_seed.py`.

Uses only Python standard library — no external dependencies required.
"""

from ..registry import TOOL_REGISTRY
from .prompt_builder import build_system_prompt as _build_system_prompt
from .combinations import generate_combinations as _generate_combinations
from .scenarios import generate_scenarios as _generate_scenarios
from .output import write_output as _write_output

__all__ = ["ToolCallingSeedGenerator", "TOOL_REGISTRY"]


class ToolCallingSeedGenerator:
    """Primary public class for generating tool-calling evaluation seed metadata.

    Encapsulates configuration and exposes methods for prompt building,
    combination generation, scenario generation, and output serialization.
    Each public method can be used independently without requiring a full
    generation pipeline execution.

    Parameters
    ----------
    num_variants : int, optional
        Number of variants produced per combination for statistical power.
        Must be a positive integer (minimum 1). Defaults to 3.
    """

    def __init__(self, num_variants: int = 3) -> None:
        """Initialize the generator with the specified number of variants.

        Parameters
        ----------
        num_variants : int, optional
            Number of variants per combination. Defaults to 3.

        Raises
        ------
        ValueError
            If num_variants is not an integer or is less than 1.
        """
        if isinstance(num_variants, bool) or not isinstance(num_variants, int):
            raise ValueError(
                f"num_variants must be a positive integer, got {type(num_variants).__name__}: {num_variants!r}"
            )
        if num_variants < 1:
            raise ValueError(
                f"num_variants must be at least 1, got {num_variants}"
            )
        self.num_variants = num_variants

    def build_system_prompt(self, enabled_tools: list[str]) -> str:
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
        return _build_system_prompt(enabled_tools)

    def generate_combinations(self) -> list[dict]:
        """Generate all non-empty subsets of the registered tools.

        Produces one combination for every non-empty subset of the current
        registry, with boolean fields per registered tool and unique
        sequential identifiers. IDs are assigned deterministically, ordered
        by subset size then registry order.

        Returns
        -------
        list[dict]
            List of combination dictionaries with combination_id,
            boolean tool fields, and tools_enabled list.
        """
        return _generate_combinations()

    def generate_scenarios(self) -> list[dict]:
        """Generate evaluation scenario metadata for all combinations and variants.

        Produces single-hop and multi-hop scenarios with golden_truth fields
        for each combination-variant pair. Each scenario includes a unique
        scenario_id, combination metadata, system prompt, expected tools,
        hop count, golden truth, and an empty questions placeholder.

        Returns
        -------
        list[dict]
            List of evaluation scenario dictionaries.
        """
        return _generate_scenarios(self.num_variants)

    def write_output(self, output_path: str = None) -> str:
        """Serialize generated scenarios to a JSON file.

        Parameters
        ----------
        output_path : str, optional
            Path to write the output file. Defaults to
            `custom_benchmarks/tool_calling/tool_calling_seed.json` relative
            to the module's location.

        Returns
        -------
        str
            Absolute path of the written file.

        Raises
        ------
        OSError
            If the output file cannot be written due to a filesystem error.
        """
        return _write_output(self.num_variants, output_path)
