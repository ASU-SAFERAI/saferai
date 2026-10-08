#!/usr/bin/env python
"""CLI script for building a compact tool-calling evaluation seed from scratch.

This script generates the full scenario pool programmatically via
``ToolCallingSeedGenerator`` (no pre-existing JSON file is read), then samples
a small, representative mini-benchmark from it. The goal is a ~60-question set
that smokes out low-performant models by emphasizing the two skills weak models
fail hardest: choosing the right tool among lookalikes (discrimination) and
correctly declining to call any tool (abstention), while still exercising
trap variety and multi-hop chains.

This is a STRUCTURE-ONLY dataset. Each scenario carries a real system prompt,
expected tools, hop count, and full metadata, but ``questions`` is left empty
and ``golden_truth`` arguments are empty placeholders (``{}``) exactly as the
generator emits them. Question text and golden-argument values must be filled
in by a later step before the set can be run against a model.

Target composition (60 scenarios):
    - discrimination:          30 (50%)
    - abstention:              18 (30%)
    - schema_fit:               6 (10%)
    - constraint_satisfaction:  6 (10%)

Selection is deterministic given a seed, so the output is reproducible.

Usage:
    python -m pre_deploy.dataset_generation.tool_calling
    python -m pre_deploy.dataset_generation.tool_calling --summary
    python -m pre_deploy.dataset_generation.tool_calling --seed 7 --num-variants 5
    python -m pre_deploy.dataset_generation.tool_calling --output-path out.json
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pre_deploy.dataset_generation.tool_calling import ToolCallingSeedGenerator


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT = _MODULE_DIR / "tool_calling_seed_60.json"


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------
# Each entry defines a "cell": a (construct_type, trap_type, hop_count) triple
# and how many scenarios to draw from it. trap_type of None means "no trap".
# hop_count of None means "any hop count is acceptable for this cell".
#
# The plan is tuned to what the pool actually contains:
#   - order_bias is the only multi-hop trap and lives only in discrimination.
#   - abstention is always hop 0 and carries eager_tool_use / name_fallacy /
#     redundant_tooling traps.
#   - schema_fit has no traps; hop variety stands in for it.
#   - constraint_satisfaction is always redundant_tooling with hops 1/2/4.
#
# Cells are matched against the pool. If a cell cannot be fully satisfied,
# the shortfall is reported and back-filled from the same construct_type so
# the per-construct quota (and total) is still met.

_SamplingCell = tuple  # (construct_type, trap_type, hop_count, count)

SAMPLING_PLAN: list[_SamplingCell] = [
    # -- discrimination: 30 --------------------------------------------------
    ("discrimination", "order_bias", 1, 2),
    ("discrimination", "order_bias", 2, 2),
    ("discrimination", "order_bias", 3, 2),
    ("discrimination", "order_bias", 4, 2),
    ("discrimination", "eager_tool_use", 1, 4),
    ("discrimination", "name_fallacy", 1, 4),
    ("discrimination", "redundant_tooling", 1, 3),
    ("discrimination", None, 1, 4),
    ("discrimination", None, 2, 3),
    ("discrimination", None, 3, 2),
    ("discrimination", None, 4, 2),
    # -- abstention: 18 (all hop 0) -----------------------------------------
    ("abstention", "eager_tool_use", 0, 7),
    ("abstention", "name_fallacy", 0, 7),
    ("abstention", "redundant_tooling", 0, 4),
    # -- schema_fit: 6 (no traps) -------------------------------------------
    ("schema_fit", None, 1, 3),
    ("schema_fit", None, 2, 2),
    ("schema_fit", None, 3, 1),
    # -- constraint_satisfaction: 6 (all redundant_tooling) -----------------
    ("constraint_satisfaction", "redundant_tooling", 1, 2),
    ("constraint_satisfaction", "redundant_tooling", 2, 2),
    ("constraint_satisfaction", "redundant_tooling", 4, 2),
]

# Per-construct quotas, derived from the plan. Used for back-fill when a
# specific (trap, hop) cell is short.
CONSTRUCT_QUOTAS: dict[str, int] = defaultdict(int)
for _construct, _trap, _hop, _count in SAMPLING_PLAN:
    CONSTRUCT_QUOTAS[_construct] += _count


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _scenario_key(s: dict) -> tuple:
    """A stable identity for a scenario, used to avoid double-selection."""
    return (s.get("scenario_id"), s.get("combination_id"), s.get("variant"))


def _matches_cell(s: dict, construct: str, trap, hop) -> bool:
    meta = s.get("metadata", {})
    if meta.get("construct_type") != construct:
        return False
    if meta.get("trap_type") != trap:
        return False
    if hop is not None and s.get("hop_count") != hop:
        return False
    return True


def sample_scenarios(
    pool: list[dict],
    seed: int,
) -> tuple[list[dict], list[str]]:
    """Draw scenarios from ``pool`` according to ``SAMPLING_PLAN``.

    Returns the selected scenarios (ordered by construct, then trap, then hop)
    and a list of human-readable warnings about any cells that fell short.
    """
    rng = random.Random(seed)
    warnings: list[str] = []
    selected: list[dict] = []
    selected_keys: set = set()
    per_construct_selected: Counter = Counter()

    def _take(candidates: list[dict], n: int) -> list[dict]:
        """Deterministically take up to n unused candidates."""
        available = [s for s in candidates if _scenario_key(s) not in selected_keys]
        # Sort for determinism before shuffling with the seeded RNG.
        available.sort(key=lambda s: str(_scenario_key(s)))
        rng.shuffle(available)
        picked = available[:n]
        for s in picked:
            selected_keys.add(_scenario_key(s))
        return picked

    # Pass 1: satisfy each explicit cell.
    for construct, trap, hop, count in SAMPLING_PLAN:
        candidates = [s for s in pool if _matches_cell(s, construct, trap, hop)]
        picked = _take(candidates, count)
        if len(picked) < count:
            warnings.append(
                f"cell (construct={construct}, trap={trap}, hop={hop}) "
                f"wanted {count} but pool had {len(picked)} available"
            )
        selected.extend(picked)
        per_construct_selected[construct] += len(picked)

    # Pass 2: back-fill any per-construct shortfall from the same construct,
    # ignoring the specific trap/hop cell so the quota (and total) is met.
    for construct, quota in CONSTRUCT_QUOTAS.items():
        shortfall = quota - per_construct_selected[construct]
        if shortfall <= 0:
            continue
        candidates = [
            s for s in pool
            if s.get("metadata", {}).get("construct_type") == construct
        ]
        picked = _take(candidates, shortfall)
        if picked:
            warnings.append(
                f"back-filled {len(picked)} scenario(s) for construct "
                f"'{construct}' from other trap/hop cells"
            )
        selected.extend(picked)
        per_construct_selected[construct] += len(picked)
        still_short = quota - per_construct_selected[construct]
        if still_short > 0:
            warnings.append(
                f"construct '{construct}' still short by {still_short} "
                f"(pool exhausted)"
            )

    # Stable output ordering for a readable, diff-friendly file.
    def _order_key(s: dict) -> tuple:
        meta = s.get("metadata", {})
        construct_order = {
            "discrimination": 0,
            "abstention": 1,
            "schema_fit": 2,
            "constraint_satisfaction": 3,
        }
        return (
            construct_order.get(meta.get("construct_type"), 99),
            str(meta.get("trap_type")),
            s.get("hop_count", 0),
            str(_scenario_key(s)),
        )

    selected.sort(key=_order_key)
    return selected, warnings


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def _print_summary(scenarios: list[dict]) -> None:
    """Print a breakdown table of scenario metadata dimensions."""
    construct_counts: Counter = Counter()
    depth_counts: Counter = Counter()
    trap_counts: Counter = Counter()
    pool_counts: Counter = Counter()
    hop_counts: Counter = Counter()
    tool_counts: Counter = Counter()
    overlap_counts: Counter = Counter()
    dependency_counts: Counter = Counter()
    distractor_n_counts: Counter = Counter()
    distractor_type_counts: Counter = Counter()
    cog_load_n_counts: Counter = Counter()
    cog_load_flag_counts: Counter = Counter()
    combo_counts: Counter = Counter()
    scenarios_with_a_tool = 0
    abstention_scenarios = 0
    scenarios_with_trap = 0

    for s in scenarios:
        meta = s.get("metadata", {})
        construct_counts[meta.get("construct_type", "unknown")] += 1
        depth_counts[meta.get("cognitive_depth", "unknown")] += 1
        trap_counts[meta.get("trap_type")] += 1
        pool_counts[meta.get("candidate_pool_size", 0)] += 1
        hop_counts[s.get("hop_count", 0)] += 1
        overlap_counts[meta.get("overlap_degree", 0)] += 1
        dependency_counts[meta.get("dependency_depth", 0)] += 1
        combo_counts[s.get("combination_id", "unknown")] += 1

        if meta.get("trap_type") is not None:
            scenarios_with_trap += 1

        distractor_types = meta.get("distractor_types", [])
        distractor_n_counts[len(distractor_types)] += 1
        for distractor in distractor_types:
            distractor_type_counts[distractor] += 1

        active_flags = [k for k, v in meta.get("cognitive_load", {}).items() if v]
        cog_load_n_counts[len(active_flags)] += 1
        for flag in active_flags:
            cog_load_flag_counts[flag] += 1

        expected_tools = s.get("expected_tools", [])
        if expected_tools:
            scenarios_with_a_tool += 1
            # Count each distinct tool once per scenario so multi-hop
            # scenarios don't over-count a tool they call repeatedly.
            for tool in set(expected_tools):
                tool_counts[tool] += 1
        else:
            abstention_scenarios += 1

    print("\n--- Summary Breakdown ---\n")

    print("By construct_type:")
    print("  def: what capability the scenario tests when selecting a tool.")
    print("    - discrimination: choosing the right tool for the capability the situation requires")
    print("    - constraint_satisfaction: choosing based on non-functional requirements")
    print("      (rate limits, latency, cost, payload size)")
    print("    - schema_fit: choosing based on required inputs/outputs")
    print("      (e.g., tool needs structured JSON vs plain text)")
    print("    - abstention: correctly deciding NOT to use any tool when none is appropriate")
    for key, count in sorted(construct_counts.items()):
        print(f"  {key}: {count}")

    print("\nBy cognitive_depth:")
    print("  def: the Bloom's taxonomy level of reasoning the scenario demands,")
    print("    from lowest to highest: remember < understand < apply < analyze < evaluate < create")
    for key, count in sorted(depth_counts.items()):
        print(f"  {key}: {count}")

    print("\nBy trap_type:")
    print("  def: a deliberately misleading condition testing failure modes (None = no trap).")
    print("    - eager_tool_use: prompt implies a simple direct answer, but tempting tools are present")
    print("    - name_fallacy: a tool's name implies functionality its description does not support")
    print("    - order_bias: correct tool placed in a non-salient position in the context window")
    print("    - redundant_tooling: multiple tools can solve the query; the model must pick the optimal one")
    for key, count in sorted(trap_counts.items(), key=lambda x: (x[0] is None, x[0] or "")):
        label = key if key is not None else "None"
        print(f"  {label}: {count}")

    print("\nBy candidate_pool_size:")
    print("  def: number of tools made available in the context window for that scenario (1-4).")
    for key, count in sorted(pool_counts.items()):
        print(f"  {key}: {count}")

    total = sum(hop_counts.values())
    print("\nBy hop_count:")
    print("  def: number of sequential tool calls required to fully resolve the query.")
    print("    0 = abstention (no tool call), 1 = single tool, 2+ = multi-step chains.")
    for key, count in sorted(hop_counts.items()):
        pct = count * 100 / total if total else 0
        print(f"  {key}: {count} ({pct:.1f}%)")

    print("\nBy tool (scenarios involving each tool call):")
    print("  def: how many scenarios call each specific tool at least once. Multi-hop")
    print("    scenarios count a repeated tool only once. Abstention scenarios call no tool.")
    print(f"  scenarios calling >= 1 tool: {scenarios_with_a_tool}")
    print(f"  abstention (no tool call): {abstention_scenarios}")
    for key, count in sorted(tool_counts.items(), key=lambda x: (-x[1], x[0])):
        pct = count * 100 / scenarios_with_a_tool if scenarios_with_a_tool else 0
        print(f"  {key}: {count} ({pct:.1f}% of tool-calling scenarios)")

    total_pct = len(scenarios) or 1

    print("\nBy overlap_degree:")
    print("  def: how similar the available tools' capabilities are, on a 1-5 scale:")
    print("    1 = mutually exclusive capabilities,")
    print("    5 = nearly identical capabilities distinguished only by edge cases.")
    for key, count in sorted(overlap_counts.items()):
        print(f"  {key}: {count} ({count * 100 / total_pct:.1f}%)")

    print("\nBy dependency_depth:")
    print("  def: number of prerequisite tool outputs that must be produced before this")
    print("    decision can be made (0 = no prerequisites).")
    for key, count in sorted(dependency_counts.items()):
        print(f"  {key}: {count} ({count * 100 / total_pct:.1f}%)")

    print("\nBy trap presence:")
    print("  def: whether the scenario includes any trap_type (see trap_type above).")
    no_trap = len(scenarios) - scenarios_with_trap
    print(f"  with trap: {scenarios_with_trap} ({scenarios_with_trap * 100 / total_pct:.1f}%)")
    print(f"  no trap: {no_trap} ({no_trap * 100 / total_pct:.1f}%)")

    print("\nBy distractor_types count per scenario:")
    print("  def: how many distractor tools of any kind are present in the scenario's pool.")
    for key, count in sorted(distractor_n_counts.items()):
        print(f"  {key}: {count} ({count * 100 / total_pct:.1f}%)")

    print("\nBy distractor_type (counted once per occurrence):")
    print("  def: the kinds of misleading tools present.")
    print("    - superficially_similar: shares keywords but has a different capability")
    print("    - over_capability: works but is inefficient, costly, or adds risk/complexity")
    print("    - under_capability: handles part of the task but misses critical edge cases")
    print("    - parameter_mismatch: performs the right action but cannot accept the prompt's data format")
    for key, count in sorted(distractor_type_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {key}: {count}")

    print("\nBy cognitive_load active-flag count per scenario:")
    print("  def: how many cognitive-load challenges are active at once in the scenario.")
    for key, count in sorted(cog_load_n_counts.items()):
        print(f"  {key}: {count} ({count * 100 / total_pct:.1f}%)")

    print("\nBy cognitive_load flag (counted once per occurrence):")
    print("  def: specific reasoning challenges present.")
    print("    - implicit_need_detection: model must infer the tool need without explicit keywords")
    print("    - multi_step_chaining: output of one tool is required as input to another")
    print("    - negation_handling: prompt explicitly forbids certain tools or parameters")
    print("    - schema_parsing: model must correctly interpret deeply nested API specs")
    for key, count in sorted(cog_load_flag_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {key}: {count} ({count * 100 / total_pct:.1f}% of scenarios)")

    print("\nBy combination_id (scenarios per tool combination):")
    print("  def: each combination_id is one specific subset of enabled tools; this shows")
    print("    how many scenarios were selected per combination.")
    if combo_counts:
        combo_values = combo_counts.values()
        print(f"  min: {min(combo_values)}  max: {max(combo_values)}  combos: {len(combo_counts)}")
    for key, count in sorted(combo_counts.items()):
        print(f"  {key}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the sampler CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Sample a compact, representative tool-calling mini-benchmark from "
            "the full seed pool."
        )
    )
    parser.add_argument(
        "--num-variants",
        type=int,
        default=1,
        help=(
            "Number of variants per tool combination the generator produces "
            "before sampling (positive integer, default: 3). Larger values "
            "give the sampler a bigger pool to draw from."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(_DEFAULT_OUTPUT),
        help=(
            "Path to write the sampled JSON file "
            f"(default: {_DEFAULT_OUTPUT.name} in the tool_calling directory)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible selection (default: 42)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a summary breakdown of the sampled scenarios",
    )

    args = parser.parse_args()

    try:
        # Generate the full scenario pool from scratch (no file read).
        generator = ToolCallingSeedGenerator(num_variants=args.num_variants)
        pool = generator.generate_scenarios()

        planned_total = sum(CONSTRUCT_QUOTAS.values())

        selected, warnings = sample_scenarios(pool, seed=args.seed)

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)

        print(f"Generated pool size: {len(pool)} (num_variants={args.num_variants})")
        print(f"Sampled scenarios: {len(selected)} (planned {planned_total})")
        print(f"Seed: {args.seed}")
        print(f"Output written to: {output_path}")

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")

        if args.summary:
            _print_summary(selected)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
