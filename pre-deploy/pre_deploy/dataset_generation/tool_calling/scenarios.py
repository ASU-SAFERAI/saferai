"""Scenario generation for tool-calling evaluation.

Provides a standalone function to generate evaluation scenario metadata
for all combinations and variants using a scenario archetype approach.
Each scenario includes 8 metadata dimensions for richer evaluation.
"""

from .combinations import generate_combinations
from .prompt_builder import build_system_prompt
from .registry import TOOL_REGISTRY

__all__ = ["generate_scenarios"]

# ---------------------------------------------------------------------------
# Overlap degree calculation helpers
# ---------------------------------------------------------------------------

# Pairwise overlap scores between specific tool pairs.
# Keys are stored alphabetically sorted to match the sorted lookup in
# _compute_overlap_degree. create_artifact belongs to the "visual output"
# cluster with generate_image and visualize_widget, so it shares their
# overlap score of 2 (same rubric as generate_image x visualize_widget).
_PAIRWISE_OVERLAP: dict[tuple[str, str], int] = {
    ("isearch_search", "websearch"): 3,
    ("generate_image", "visualize_widget"): 2,
    ("create_artifact", "generate_image"): 2,
    ("create_artifact", "visualize_widget"): 3,
}


def _compute_overlap_degree(tools_enabled: list[str]) -> int:
    """Compute the max pairwise overlap among enabled tools.

    Uses predefined pairwise overlap scores. All pairs not explicitly
    listed default to overlap 1.
    """
    if len(tools_enabled) <= 1:
        return 1

    max_overlap = 1
    for i, t1 in enumerate(tools_enabled):
        for t2 in tools_enabled[i + 1:]:
            pair = tuple(sorted([t1, t2]))
            overlap = _PAIRWISE_OVERLAP.get(pair, 1)
            if overlap > max_overlap:
                max_overlap = overlap
    return max_overlap


# ---------------------------------------------------------------------------
# Cognitive load helper
# ---------------------------------------------------------------------------

def _cognitive_load(
    implicit_need_detection: bool = False,
    multi_step_chaining: bool = False,
    negation_handling: bool = False,
    schema_parsing: bool = False,
) -> dict[str, bool]:
    """Build a cognitive_load dict with explicit boolean flags."""
    return {
        "implicit_need_detection": implicit_need_detection,
        "multi_step_chaining": multi_step_chaining,
        "negation_handling": negation_handling,
        "schema_parsing": schema_parsing,
    }


# ---------------------------------------------------------------------------
# Tools with complex/optional parameters (for schema_fit archetype)
# ---------------------------------------------------------------------------


def _has_complex_parameters(definition: dict) -> bool:
    """Identify schema-fit tools from registry metadata, not tool names."""
    parameters = [
        *definition.get("required_parameters", []),
        *definition.get("optional_parameters", []),
    ]
    return bool(definition.get("optional_parameters")) or any(
        "sections" in str(parameter.get("constraints", "")).lower()
        for parameter in parameters
    )


_COMPLEX_PARAM_TOOLS = frozenset(
    tool_name
    for tool_name, definition in TOOL_REGISTRY.items()
    if _has_complex_parameters(definition)
)

# Distractor type constants
_DISTRACTOR_TYPES = [
    "superficially_similar",
    "over_capability",
    "under_capability",
    "parameter_mismatch",
]

# Trap types
_TRAP_TYPES = ["eager_tool_use", "name_fallacy", "order_bias", "redundant_tooling"]


# ---------------------------------------------------------------------------
# Blacklist / Filtering Rules
# ---------------------------------------------------------------------------

def _passes_blacklist(metadata: dict, candidate_pool_size: int) -> bool:
    """Return True if the scenario passes all blacklist filters.

    Discards scenarios that are contradictory or low-value.
    """
    construct_type = metadata["construct_type"]
    cognitive_load = metadata["cognitive_load"]
    overlap_degree = metadata["overlap_degree"]
    distractor_types = metadata["distractor_types"]
    trap_type = metadata["trap_type"]
    cognitive_depth = metadata["cognitive_depth"]
    dependency_depth = metadata["dependency_depth"]

    # Rule 1: abstention + multi_step_chaining
    if construct_type == "abstention" and cognitive_load["multi_step_chaining"]:
        return False

    # Rule 2: abstention + dependency_depth > 0
    if construct_type == "abstention" and dependency_depth > 0:
        return False

    # Rule 3: overlap_degree >= 4 + candidate_pool_size == 1
    if overlap_degree >= 4 and candidate_pool_size == 1:
        return False

    # Rule 4: distractor_types non-empty + candidate_pool_size == 1
    if distractor_types and candidate_pool_size == 1:
        return False

    # Rule 5: trap_type == "order_bias" + candidate_pool_size <= 2
    if trap_type == "order_bias" and candidate_pool_size <= 2:
        return False

    # Rule 6: trap_type == "redundant_tooling" + candidate_pool_size < 3
    if trap_type == "redundant_tooling" and candidate_pool_size < 3:
        return False

    # Rule 7: cognitive_depth == "remember" + any non-null trap_type
    if cognitive_depth == "remember" and trap_type is not None:
        return False

    return True


# ---------------------------------------------------------------------------
# Archetype generators
# ---------------------------------------------------------------------------

def _archetype_basic_discrimination(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype A: Basic Discrimination — one scenario per enabled tool (single-hop)."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    for tool_key in tools_enabled:
        metadata = {
            "construct_type": "discrimination",
            "cognitive_depth": "apply",
            "candidate_pool_size": pool_size,
            "overlap_degree": min(overlap_degree, 2),  # 1-2 for basic
            "distractor_types": [],
            "cognitive_load": _cognitive_load(),
            "trap_type": None,
            "dependency_depth": 0,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = f"{combination_id}_v{variant}_discrimination_basic_{tool_key}"
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": 1,
            "expected_tools": [tool_key],
            "golden_truth": [{"tool_name": tool_key, "arguments": {}}],
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


def _archetype_distractor_resistance(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype B: Distractor Resistance — one sub-scenario per distractor type."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # Only applies to combinations with 2+ tools
    if pool_size < 2:
        return []

    # Rotate the target across variants so involvement isn't skewed toward
    # the earliest registry tool.
    ordered_enabled = [k for k in registry_keys if k in tools_enabled]
    target_tool = _rotate(ordered_enabled, variant)[0]

    for distractor_type in _DISTRACTOR_TYPES:
        metadata = {
            "construct_type": "discrimination",
            "cognitive_depth": "analyze",
            "candidate_pool_size": pool_size,
            "overlap_degree": min(max(overlap_degree, 3), 4),  # 3-4
            "distractor_types": [distractor_type],
            "cognitive_load": _cognitive_load(implicit_need_detection=True),
            "trap_type": None,
            "dependency_depth": 0,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = f"{combination_id}_v{variant}_distractor_{distractor_type}"
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": 1,
            "expected_tools": [target_tool],
            "golden_truth": [{"tool_name": target_tool, "arguments": {}}],
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


def _archetype_schema_fit(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype C: Schema Fit Assessment — for tools with complex params."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # Only applies to tools with optional/complex params
    complex_tools_in_combo = [t for t in tools_enabled if t in _COMPLEX_PARAM_TOOLS]
    if not complex_tools_in_combo:
        return []

    for tool_key in complex_tools_in_combo:
        metadata = {
            "construct_type": "schema_fit",
            "cognitive_depth": "analyze",
            "candidate_pool_size": pool_size,
            "overlap_degree": min(max(overlap_degree, 2), 3),  # 2-3
            "distractor_types": ["parameter_mismatch"],
            "cognitive_load": _cognitive_load(schema_parsing=True),
            "trap_type": None,
            "dependency_depth": 0,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = f"{combination_id}_v{variant}_schema_fit_{tool_key}"
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": 1,
            "expected_tools": [tool_key],
            "golden_truth": [{"tool_name": tool_key, "arguments": {}}],
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


def _archetype_constraint_satisfaction(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype D: Constraint Satisfaction — 3+ tools where multiple could work."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # Only applies to combinations with 3+ tools
    if pool_size < 3:
        return []

    # Rotate the target across variants to balance per-tool involvement.
    ordered_enabled = [k for k in registry_keys if k in tools_enabled]
    target_tool = _rotate(ordered_enabled, variant)[0]

    metadata = {
        "construct_type": "constraint_satisfaction",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": min(max(overlap_degree, 3), 4),  # 3-4
        "distractor_types": ["over_capability"],
        "cognitive_load": _cognitive_load(implicit_need_detection=True),
        "trap_type": "redundant_tooling",
        "dependency_depth": 0,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_constraint_satisfaction"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 1,
        "expected_tools": [target_tool],
        "golden_truth": [{"tool_name": target_tool, "arguments": {}}],
        "questions": [],
        "metadata": metadata,
    }
    scenarios.append(scenario)

    return scenarios


def _archetype_abstention(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype E: Abstention Required — correct answer is to NOT call any tool."""
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    metadata = {
        "construct_type": "abstention",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 2,
        "distractor_types": ["superficially_similar", "under_capability"],
        "cognitive_load": _cognitive_load(negation_handling=True),
        "trap_type": "eager_tool_use",
        "dependency_depth": 0,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_abstention"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 0,
        "expected_tools": [],
        "golden_truth": [],
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_multi_step_chaining(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype F: Multi-Step Chaining — multi-hop scenarios."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    n_tools = len(tools_enabled)
    if n_tools < 2:
        return []

    # Order enabled tools by registry position, then rotate by variant so the
    # chained prefix isn't always anchored on the earliest registry tool.
    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )

    for hop_count in range(2, n_tools + 1):
        expected_tools = ordered_enabled[:hop_count]
        golden_truth = [
            {"tool_name": t, "arguments": {}} for t in expected_tools
        ]

        metadata = {
            "construct_type": "discrimination",
            "cognitive_depth": "create",
            "candidate_pool_size": pool_size,
            "overlap_degree": min(overlap_degree, 2),  # 1-2
            "distractor_types": [],
            "cognitive_load": _cognitive_load(multi_step_chaining=True),
            "trap_type": None,
            "dependency_depth": hop_count - 1,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = f"{combination_id}_v{variant}_chaining_{hop_count}hop"
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": hop_count,
            "expected_tools": expected_tools,
            "golden_truth": golden_truth,
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


def _archetype_trap_scenarios(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype G: Trap Scenarios — one per trap type."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # Only applies to combinations with 2+ tools
    if pool_size < 2:
        return []

    # Rotate the target across variants to balance per-tool involvement.
    ordered_enabled = [k for k in registry_keys if k in tools_enabled]
    target_tool = _rotate(ordered_enabled, variant)[0]

    trap_configs = {
        "eager_tool_use": {
            "overlap_degree": 3,
            "distractor_types": ["superficially_similar"],
            "cognitive_load": _cognitive_load(implicit_need_detection=True),
        },
        "name_fallacy": {
            "overlap_degree": 4,
            "distractor_types": ["superficially_similar"],
            "cognitive_load": _cognitive_load(implicit_need_detection=True),
        },
        "order_bias": {
            "overlap_degree": 3,
            "distractor_types": ["over_capability"],
            "cognitive_load": _cognitive_load(schema_parsing=True),
        },
        "redundant_tooling": {
            "overlap_degree": 5,
            "distractor_types": ["over_capability", "superficially_similar"],
            "cognitive_load": _cognitive_load(
                implicit_need_detection=True, schema_parsing=True,
            ),
        },
    }

    for trap_type, config in trap_configs.items():
        metadata = {
            "construct_type": "discrimination",
            "cognitive_depth": "evaluate",
            "candidate_pool_size": pool_size,
            "overlap_degree": config["overlap_degree"],
            "distractor_types": config["distractor_types"],
            "cognitive_load": config["cognitive_load"],
            "trap_type": trap_type,
            "dependency_depth": 0,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = f"{combination_id}_v{variant}_trap_{trap_type}"
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": 1,
            "expected_tools": [target_tool],
            "golden_truth": [{"tool_name": target_tool, "arguments": {}}],
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


def _archetype_implicit_need_detection(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype H: Implicit Need Detection — single-tool scenarios."""
    scenarios = []
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    for tool_key in tools_enabled:
        metadata = {
            "construct_type": "discrimination",
            "cognitive_depth": "understand",
            "candidate_pool_size": pool_size,
            "overlap_degree": 2,
            "distractor_types": ["superficially_similar"],
            "cognitive_load": _cognitive_load(implicit_need_detection=True),
            "trap_type": None,
            "dependency_depth": 0,
        }
        if not _passes_blacklist(metadata, pool_size):
            continue

        scenario_id = (
            f"{combination_id}_v{variant}_implicit_need_{tool_key}"
        )
        scenario = {
            "scenario_id": scenario_id,
            "combination_id": combination_id,
            "variant": variant,
            "enabled_tools": enabled_tools_map,
            "system_prompt": system_prompt,
            "hop_count": 1,
            "expected_tools": [tool_key],
            "golden_truth": [{"tool_name": tool_key, "arguments": {}}],
            "questions": [],
            "metadata": metadata,
        }
        scenarios.append(scenario)

    return scenarios


# ---------------------------------------------------------------------------
# New Abstention Archetypes (boost hop_count=0)
# ---------------------------------------------------------------------------

def _archetype_abstention_name_fallacy(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype I: Abstention Name Fallacy — abstain despite name-similar tools."""
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    metadata = {
        "construct_type": "abstention",
        "cognitive_depth": "analyze",
        "candidate_pool_size": pool_size,
        "overlap_degree": 3,
        "distractor_types": ["superficially_similar"],
        "cognitive_load": _cognitive_load(implicit_need_detection=True),
        "trap_type": "name_fallacy",
        "dependency_depth": 0,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_abstention_name_fallacy"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 0,
        "expected_tools": [],
        "golden_truth": [],
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_abstention_constraint(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype J: Abstention Constraint — abstain when constraints unsatisfiable."""
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    metadata = {
        "construct_type": "abstention",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 4,
        "distractor_types": ["over_capability", "parameter_mismatch"],
        "cognitive_load": _cognitive_load(schema_parsing=True, negation_handling=True),
        "trap_type": "redundant_tooling",
        "dependency_depth": 0,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_abstention_constraint"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 0,
        "expected_tools": [],
        "golden_truth": [],
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


# ---------------------------------------------------------------------------
# New 2-hop Archetypes (boost hop_count=2)
# ---------------------------------------------------------------------------

def _archetype_chaining_with_distractors(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype K: 2-hop chaining with distractors."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    if pool_size < 2:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:2]
    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "analyze",
        "candidate_pool_size": pool_size,
        "overlap_degree": min(max(overlap_degree, 2), 3),
        "distractor_types": ["over_capability"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, implicit_need_detection=True,
        ),
        "trap_type": None,
        "dependency_depth": 1,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_distractor_2hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 2,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_schema_fit(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype L: 2-hop chaining with schema fit assessment."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    if pool_size < 2:
        return []

    # Only applies when at least one complex-param tool is enabled
    complex_tools_in_combo = [t for t in tools_enabled if t in _COMPLEX_PARAM_TOOLS]
    if not complex_tools_in_combo:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:2]
    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "schema_fit",
        "cognitive_depth": "create",
        "candidate_pool_size": pool_size,
        "overlap_degree": 2,
        "distractor_types": ["parameter_mismatch"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, schema_parsing=True,
        ),
        "trap_type": None,
        "dependency_depth": 1,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_schema_2hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 2,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_reverse_order(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype M: 2-hop chaining with reverse tool order (tests order bias)."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # order_bias trap requires pool > 2
    if pool_size <= 2:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    # Select LAST 2 tools of the (rotated) ordering to exercise order bias
    expected_tools = ordered_enabled[-2:]
    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 3,
        "distractor_types": ["superficially_similar"],
        "cognitive_load": _cognitive_load(multi_step_chaining=True),
        "trap_type": "order_bias",
        "dependency_depth": 1,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_reverse_2hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 2,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_constraint_2hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype N: 2-hop chaining with constraint satisfaction."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # redundant_tooling trap requires pool >= 3
    if pool_size < 3:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:2]
    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "constraint_satisfaction",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 4,
        "distractor_types": ["over_capability"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, implicit_need_detection=True,
        ),
        "trap_type": "redundant_tooling",
        "dependency_depth": 1,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_constraint_2hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 2,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


# ---------------------------------------------------------------------------
# New 3-hop Archetypes (boost hop_count=3)
# ---------------------------------------------------------------------------

def _archetype_chaining_trap_3hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype O: 3-hop chaining with order bias trap."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # order_bias needs pool > 2
    if pool_size <= 2:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:3]
    if len(expected_tools) < 3:
        return []

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 3,
        "distractor_types": ["superficially_similar", "over_capability"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, implicit_need_detection=True,
        ),
        "trap_type": "order_bias",
        "dependency_depth": 2,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_trap_3hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 3,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_negation_3hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype P: 3-hop chaining with negation handling."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    if pool_size < 3:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:3]
    if len(expected_tools) < 3:
        return []

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 2,
        "distractor_types": ["under_capability"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, negation_handling=True,
        ),
        "trap_type": None,
        "dependency_depth": 2,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_negation_3hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 3,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_reverse_3hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype Q: 3-hop chaining in reversed order (order bias)."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # order_bias needs pool > 2
    if pool_size <= 2:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = list(reversed(ordered_enabled[:3]))
    if len(expected_tools) < 3:
        return []

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "create",
        "candidate_pool_size": pool_size,
        "overlap_degree": 3,
        "distractor_types": [],
        "cognitive_load": _cognitive_load(multi_step_chaining=True),
        "trap_type": "order_bias",
        "dependency_depth": 2,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_reverse_3hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 3,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_schema_3hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype R: 3-hop chaining with schema fit assessment."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    if pool_size < 3:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )
    expected_tools = ordered_enabled[:3]
    if len(expected_tools) < 3:
        return []

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "schema_fit",
        "cognitive_depth": "create",
        "candidate_pool_size": pool_size,
        "overlap_degree": 2,
        "distractor_types": ["parameter_mismatch"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, schema_parsing=True,
        ),
        "trap_type": None,
        "dependency_depth": 2,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_schema_3hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 3,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


# ---------------------------------------------------------------------------
# New 4-hop Archetypes (boost hop_count=4)
# ---------------------------------------------------------------------------

def _archetype_chaining_full_4hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype S: Full 4-hop chaining (with tool reuse for 3-tool combos)."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    if pool_size < 3:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )

    if pool_size >= 4:
        expected_tools = ordered_enabled[:4]
    else:
        # 3-tool combos: reuse first tool
        expected_tools = ordered_enabled[:3] + [ordered_enabled[0]]

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "create",
        "candidate_pool_size": pool_size,
        "overlap_degree": overlap_degree,
        "distractor_types": ["over_capability", "superficially_similar"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, implicit_need_detection=True,
        ),
        "trap_type": None,
        "dependency_depth": 3,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_full_4hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 4,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_trap_4hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype T: 4-hop chaining with order bias trap (reversed order)."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # order_bias needs pool > 2
    if pool_size <= 2:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )

    if pool_size >= 4:
        expected_tools = list(reversed(ordered_enabled[:4]))
    else:
        # 3-tool combos: reversed + reuse last (which was first before reverse)
        reversed_tools = list(reversed(ordered_enabled[:3]))
        expected_tools = reversed_tools + [reversed_tools[-1]]

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "discrimination",
        "cognitive_depth": "evaluate",
        "candidate_pool_size": pool_size,
        "overlap_degree": 3,
        "distractor_types": ["superficially_similar"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, schema_parsing=True,
        ),
        "trap_type": "order_bias",
        "dependency_depth": 3,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_trap_4hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 4,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


def _archetype_chaining_constraint_4hop(
    combo: dict, variant: int, system_prompt: str, overlap_degree: int,
    pool_size: int, registry_keys: list[str],
) -> list[dict]:
    """Archetype U: 4-hop chaining with constraint satisfaction."""
    tools_enabled = combo["tools_enabled"]
    combination_id = combo["combination_id"]
    enabled_tools_map = _enabled_tools_map(combo)

    # redundant_tooling needs pool >= 3
    if pool_size < 3:
        return []

    ordered_enabled = _rotate(
        [k for k in registry_keys if k in tools_enabled], variant
    )

    if pool_size >= 4:
        expected_tools = ordered_enabled[:4]
    else:
        # 3-tool combos: reuse middle tool
        expected_tools = ordered_enabled[:3] + [ordered_enabled[1]]

    golden_truth = [{"tool_name": t, "arguments": {}} for t in expected_tools]

    metadata = {
        "construct_type": "constraint_satisfaction",
        "cognitive_depth": "create",
        "candidate_pool_size": pool_size,
        "overlap_degree": 4,
        "distractor_types": ["parameter_mismatch", "over_capability"],
        "cognitive_load": _cognitive_load(
            multi_step_chaining=True, schema_parsing=True, negation_handling=True,
        ),
        "trap_type": "redundant_tooling",
        "dependency_depth": 3,
    }
    if not _passes_blacklist(metadata, pool_size):
        return []

    scenario_id = f"{combination_id}_v{variant}_chaining_constraint_4hop"
    scenario = {
        "scenario_id": scenario_id,
        "combination_id": combination_id,
        "variant": variant,
        "enabled_tools": enabled_tools_map,
        "system_prompt": system_prompt,
        "hop_count": 4,
        "expected_tools": expected_tools,
        "golden_truth": golden_truth,
        "questions": [],
        "metadata": metadata,
    }
    return [scenario]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _rotate(ordered_tools: list[str], variant: int) -> list[str]:
    """Rotate a registry-ordered tool list by the variant index.

    Archetypes that select a single target (``[0]``) or a registry-order
    prefix (``[:n]``) otherwise always favor tools that appear earliest in
    ``TOOL_REGISTRY``, skewing per-tool involvement toward the first tools.
    Rotating by ``(variant - 1)`` cycles the starting tool across variants
    deterministically, evening out which tools get targeted while keeping
    the total scenario count and per-variant structure unchanged.

    Parameters
    ----------
    ordered_tools : list[str]
        Enabled tools in canonical registry order.
    variant : int
        1-based variant index.

    Returns
    -------
    list[str]
        The list rotated left by ``(variant - 1) % len(ordered_tools)``.
    """
    if not ordered_tools:
        return ordered_tools
    offset = (variant - 1) % len(ordered_tools)
    return ordered_tools[offset:] + ordered_tools[:offset]


def _enabled_tools_map(combo: dict) -> dict[str, bool]:
    """Extract the boolean enabled-tools map from a combination dict.

    Driven by TOOL_REGISTRY order so the map stays correct as tools are
    added to or removed from the registry.
    """
    return {key: combo[key] for key in TOOL_REGISTRY}


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def generate_scenarios(num_variants: int) -> list[dict]:
    """Generate evaluation scenario metadata for all combinations and variants.

    Uses a scenario archetype approach to produce meaningful combinations
    of 8 metadata dimensions rather than a Cartesian product. Each scenario
    includes a ``metadata`` field with construct_type, cognitive_depth,
    candidate_pool_size, overlap_degree, distractor_types, cognitive_load,
    trap_type, and dependency_depth.

    Parameters
    ----------
    num_variants : int
        Number of variants produced per combination for statistical power.

    Returns
    -------
    list[dict]
        List of evaluation scenario dictionaries.
    """
    combinations = generate_combinations()
    registry_keys = list(TOOL_REGISTRY.keys())

    # All archetype generators
    archetype_fns = [
        _archetype_basic_discrimination,
        _archetype_distractor_resistance,
        _archetype_schema_fit,
        _archetype_constraint_satisfaction,
        _archetype_abstention,
        _archetype_multi_step_chaining,
        _archetype_trap_scenarios,
        _archetype_implicit_need_detection,
        # New archetypes for hop-count distribution balancing
        _archetype_abstention_name_fallacy,
        _archetype_abstention_constraint,
        _archetype_chaining_with_distractors,
        _archetype_chaining_schema_fit,
        _archetype_chaining_reverse_order,
        _archetype_chaining_constraint_2hop,
        _archetype_chaining_trap_3hop,
        _archetype_chaining_negation_3hop,
        _archetype_chaining_reverse_3hop,
        _archetype_chaining_schema_3hop,
        _archetype_chaining_full_4hop,
        _archetype_chaining_trap_4hop,
        _archetype_chaining_constraint_4hop,
    ]

    scenarios: list[dict] = []

    for combo in combinations:
        tools_enabled = combo["tools_enabled"]
        pool_size = len(tools_enabled)
        system_prompt = build_system_prompt(tools_enabled)
        overlap_degree = _compute_overlap_degree(tools_enabled)

        for variant in range(1, num_variants + 1):
            for archetype_fn in archetype_fns:
                new_scenarios = archetype_fn(
                    combo=combo,
                    variant=variant,
                    system_prompt=system_prompt,
                    overlap_degree=overlap_degree,
                    pool_size=pool_size,
                    registry_keys=registry_keys,
                )
                scenarios.extend(new_scenarios)

    return scenarios
