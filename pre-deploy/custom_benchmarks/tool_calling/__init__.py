"""Tool-calling evaluation data package.

Provides the ToolCallingSeedGenerator class and TOOL_REGISTRY constant
for generating structured evaluation metadata for tool-calling
discrimination testing.
"""

from .models import (
    CanonicalCall,
    GoldenCall,
    ScenarioView,
    TRACE_STATUSES,
    TraceStatus,
    ToolCallObservation,
)
from .seed_generation import ToolCallingSeedGenerator
from .registry import TOOL_REGISTRY
from .catalog import project_tool_catalog
from .serialization import (
    CANONICAL_TRACE_SCHEMA_VERSION,
    canonical_expected_trace,
    canonical_json,
    canonical_observed_trace,
    expected_trace_payload,
    observed_trace_payload,
    serialize_expected_trace,
    serialize_observed_trace,
)

__all__ = [
    "CanonicalCall",
    "GoldenCall",
    "ScenarioView",
    "TRACE_STATUSES",
    "TraceStatus",
    "ToolCallObservation",
    "ToolCallingSeedGenerator",
    "TOOL_REGISTRY",
    "project_tool_catalog",
    "CANONICAL_TRACE_SCHEMA_VERSION",
    "canonical_expected_trace",
    "canonical_json",
    "canonical_observed_trace",
    "expected_trace_payload",
    "observed_trace_payload",
    "serialize_expected_trace",
    "serialize_observed_trace",
]
