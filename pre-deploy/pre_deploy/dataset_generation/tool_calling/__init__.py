"""Tool-calling seed-generation subsystem.

Authors the tool-calling benchmark's scenario seed offline, using only the
Python standard library. Exposes :class:`ToolCallingSeedGenerator` and the
canonical ``TOOL_REGISTRY`` plus the focused helpers each generation stage
builds on (combination enumeration, system-prompt assembly, scenario
generation, and JSON output).

The compact-benchmark sampler CLI lives alongside this package and is run with
``python -m pre_deploy.dataset_generation.tool_calling``.
"""

from .registry import TOOL_REGISTRY
from .generator import ToolCallingSeedGenerator
from .combinations import generate_combinations
from .prompt_builder import build_system_prompt
from .scenarios import generate_scenarios
from .output import write_output

__all__ = [
    "TOOL_REGISTRY",
    "ToolCallingSeedGenerator",
    "generate_combinations",
    "build_system_prompt",
    "generate_scenarios",
    "write_output",
]
