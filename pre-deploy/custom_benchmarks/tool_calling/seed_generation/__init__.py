"""Seed-generation subsystem for the tool-calling benchmark.

This subpackage authors the benchmark's scenario seed offline. It is used only
by the ``generate_seed.py`` CLI, not by the live benchmark runtime, which reads
a pre-built seed JSON. Keeping it isolated here documents that boundary and
keeps the runtime import surface lean.
"""

from .generator import ToolCallingSeedGenerator
from .combinations import generate_combinations
from .prompt_builder import build_system_prompt
from .scenarios import generate_scenarios
from .output import write_output

__all__ = [
    "ToolCallingSeedGenerator",
    "generate_combinations",
    "build_system_prompt",
    "generate_scenarios",
    "write_output",
]
