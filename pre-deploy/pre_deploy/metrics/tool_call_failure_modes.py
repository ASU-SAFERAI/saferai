"""Tool-calling failure-mode metric.

Unlike ``accuracy_with_reference`` (a G-Eval metric that collapses to a single
scalar), this metric asks the judge model for a *structured* assessment of an
ordered tool-calling trace. In addition to an ``overall_score`` (0-1, aligned
with the golden truth) the judge returns six binary quality dimensions. For
every dimension, 1 is good and 0 is bad:

    - tool_correctness     : 1 if the ordered selected tool sequence matches the golden trace.
    - tool_potentially_valid: 1 if the ordered tool choices are plausible responses to the
                             request, 0 only if they are completely implausible (tools
                             the model should never have considered). This leaves wiggle
                             room for a reasonable sequence that differs from the golden
                             truth.
    - json_syntax          : 1 if the output was syntactically valid JSON only
                             (no prose, fences, or surrounding text).
    - required_args_present: 1 if every argument the system prompt schema marks
                             as required was included.
    - optional_args_present: 1 if every NON-DEFAULT optional argument present in
                             the golden truth was also included (omitting a
                             default-valued optional argument is not penalized),
                             AND the response did not add extra/deviating
                             non-default optional arguments that reduce
                             efficiency or correctness.
    - placeholder_propagation: 1 if every symbolic prior-output dependency in the
                             golden trace is satisfied by the corresponding observed
                             tool output, or 1 when no dependency applies.

The metric follows the same staged lifecycle as ``accuracy_with_reference``:

    start_tool_call_failure_modes(...)      # enqueue judge prompts, return now
    finalize_tool_call_failure_modes(...)   # poll; status dict until complete,
                                            # then a MetricsResults

It deliberately does NOT use a DAG: a single structured judge call per scenario
produces the whole payload, validated with a Pydantic schema through
``DeepEvalClient.batch_generate(prompts, schema=...)``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field

from ..input import EvalDataset
from ..loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ..output import MetricsResults
from ..query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from .utils import _build_metric_model, _finalize_phase_or_status

logger = logging.getLogger(__name__)

_METRIC_NAME = "tool_call_failure_modes"
_PHASE = "eval"

# The six binary dimensions, in report order. Kept as a module constant so the
# benchmark reporter and any downstream consumer can rely on a stable ordering.
DIMENSION_KEYS: List[str] = [
    "tool_correctness",
    "tool_potentially_valid",
    "json_syntax",
    "required_args_present",
    "optional_args_present",
    "placeholder_propagation",
]


class ToolCallFailureModes(BaseModel):
    """Structured judge verdict for an ordered tool-calling trace."""

    overall_score: float = Field(
        ...,
        description="Score from 0 to 1 measuring alignment with the golden truth.",
    )
    tool_correctness: int = Field(
        ...,
        description="1 (good) if the ordered selected tool sequence exactly matches the "
        "golden trace (or, for abstention scenarios, no tool was called as expected); "
        "else 0 (bad).",
    )
    tool_potentially_valid: int = Field(
        ...,
        description="1 (good) if the ordered tool choices are plausible responses to the "
        "request given the available tools; 0 (bad) only if the choices are completely "
        "implausible (tools the model should never have considered). This is intentionally "
        "lenient: a sequence that does not match the golden truth can still be a suboptimal "
        "but 'good enough' option. Reserve 0 for choices that make no sense for the request "
        "at all.",
    )
    json_syntax: int = Field(
        ...,
        description="1 (good) if the output was syntactically valid JSON only, with no "
        "prose, markdown fences, or surrounding text; else 0 (bad).",
    )
    required_args_present: int = Field(
        ...,
        description="1 (good) if the model response includes every argument the system "
        "prompt schema marks as REQUIRED for each chosen tool in the ordered trace, "
        "regardless of the golden truth; 0 (bad) if any required argument is omitted. "
        "This is presence-only: a required argument that is present but has a wrong value "
        "still scores 1 here (its incorrectness is reflected in tool_correctness and "
        "overall_score).",
    )
    optional_args_present: int = Field(
        ...,
        description="1 (good) if, for every OPTIONAL argument (per the system prompt "
        "schema) that appears in the golden truth with a NON-DEFAULT value, the model "
        "response includes that argument AND matches its golden-truth value; 0 (bad) if "
        "any such non-default optional argument is omitted or given a wrong value. "
        "Also 0 (bad) if the response ADDS optional arguments that are not in the golden "
        "truth and are NOT the documented default in a way that reduces efficiency or "
        "correctness (adding an extra optional argument set to its documented default is "
        "harmless and does not lower this score). Optional arguments absent from the golden "
        "truth, or whose golden-truth value equals the documented default, do not otherwise "
        "affect this score.",
    )
    placeholder_propagation: int = Field(
        ...,
        description="1 (good) when every {{steps[N].output}} dependency in the ordered "
        "golden trace is incorporated faithfully into the consuming observed call, or "
        "when no dependency applies; 0 (bad) for an omitted, wrong-step, unrelated, "
        "current-step, or future-step dependency.",
    )
    reason: str = Field(
        default="",
        description="Concise justification for the score and dimension flags.",
    )

    def dimensions(self) -> Dict[str, int]:
        return {key: int(getattr(self, key)) for key in DIMENSION_KEYS}


# ---------------------------------------------------------------------------
# Lightweight metric object
# ---------------------------------------------------------------------------

class _FailureModeResult:
    """Minimal metric-like object compatible with ``MetricsResults``.

    ``MetricsResults`` only reads ``.name``, ``.score``, ``.reason``,
    ``.success`` (and, via the extended ``to_dict``, ``.dimensions``). Using a
    small purpose-built object avoids depending on DeepEval's metric internals
    for this bespoke, structured metric.
    """

    def __init__(self, name: str, threshold: float) -> None:
        self.name = name
        self.threshold = threshold
        self.score: float = 0.0
        self.reason: str = "not_evaluated"
        self.success: bool = False
        self.dimensions: Dict[str, int] = {key: 0 for key in DIMENSION_KEYS}

    def apply(self, verdict: ToolCallFailureModes) -> None:
        self.score = float(verdict.overall_score)
        self.reason = verdict.reason or ""
        self.success = self.score >= self.threshold
        self.dimensions = verdict.dimensions()

    def apply_parse_error(self, raw: Any) -> None:
        self.score = 0.0
        self.reason = "error_parsing_response"
        self.success = False
        self.dimensions = {key: 0 for key in DIMENSION_KEYS}
        logger.error(
            "Error parsing failure-mode judge response: %s. Defaulting to score 0.",
            raw,
        )


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_INSTRUCTIONS = """You evaluate a tool-calling response represented as an ordered trace. Inputs:
- SYSTEM PROMPT: the tool catalog and output rules the model was told to follow.
- USER REQUEST: what the model was asked.
- GOLDEN TRUTH: the correct ordered tool-call trace, or an empty trace when no tool
  should be called.
- MODEL RESPONSE: the ordered tool-call trace the model produced.

Both trace values are canonical JSON objects with a `tool_calls` array. Compare calls by
array position: preserve cardinality, order, repeated tools, tool identity, and ordinary
argument values. The MODEL RESPONSE call objects may also contain `ordinal` and
`tool_output`; use each preceding call's `tool_output` as evidence for dependencies.
Final natural-language response prose is not part of the primary comparison.

Return exactly this JSON object and nothing else (1 = good, 0 = bad):

{{
  "overall_score": <float 0..1>,
  "tool_correctness": <0 or 1>,
  "tool_potentially_valid": <0 or 1>,
  "json_syntax": <0 or 1>,
  "required_args_present": <0 or 1>,
  "optional_args_present": <0 or 1>,
  "placeholder_propagation": <0 or 1>,
  "reason": "<brief justification>"
}}

Note on "default-valued" optional arguments: an optional argument is default-valued when
the SYSTEM PROMPT documents a default (e.g. "limit (default: 5)") and the value equals it.
Setting one to its default and omitting it are equivalent; never penalize either.

Score each field independently:

overall_score - Alignment with the complete ordered GOLDEN TRUTH. Correct tool sequence
and correct arguments trends to 1.0. Missing, extra, duplicated, reordered, or wrong-tool
calls trend to 0.0. Ignore omitted default-valued optional args. Do not reduce this score
solely because a symbolic placeholder was resolved to concrete content when the dependency
is otherwise satisfied.

tool_correctness - 1 only if the MODEL RESPONSE has the same ordered call cardinality,
tool identities, and abstention shape as the GOLDEN TRUTH. For an abstention, the model
must call no tool. Otherwise, a missing, extra, duplicated, reordered, or wrong tool makes
this 0.

tool_potentially_valid - Consider the ordered tool choices as a whole. Score 1 if the
choices are plausible for this request even when they are not the golden sequence; score 0
only when a choice makes no sense here (for example, a weather tool for sending an email).
For abstention, 1 if declining is plausible. Ignore argument-level issues.

json_syntax - 1 if the MODEL RESPONSE is valid JSON and ONLY JSON (no prose, no code fences,
no surrounding text). The canonical trace must remain the evaluated response.

required_args_present - 1 if every argument the SYSTEM PROMPT marks required for each
chosen tool is present; 0 if any is missing. Presence only: a required arg with a wrong
value still scores 1 here. If no tool was called, score 1.

optional_args_present - 1 only if BOTH hold: (a) every optional arg the GOLDEN TRUTH sets to
a NON-DEFAULT value is present with that value, and (b) the response adds no extra
non-default optional arg that hurts efficiency or correctness. Default-valued optional args
(added or omitted) never affect this. If no tool was called, score 1.

placeholder_propagation - Inspect every symbolic reference matching
`{{{{steps[N].output}}}}` in GOLDEN TRUTH arguments. N is zero-based and identifies the
preceding golden call whose output is needed by the consuming call; a valid reference must
point strictly to an earlier call. A current-step or future-step reference is invalid.
Resolve the dependency against MODEL RESPONSE `tool_calls[N].tool_output`, while also
checking that the corresponding consuming call is present and in the correct position.
Do NOT require the MODEL RESPONSE argument to contain the literal placeholder string.
Score 1 when the prior output is incorporated faithfully, including direct insertion,
quoting, JSON serialization, structured embedding, or a semantically faithful
summarization. Score 0 for an omitted dependency, the wrong predecessor output, an
unrelated substituted value, an absent consuming call, or an invalid current/future-step
reference. Score 1 for single-hop and abstention traces, where no dependency applies.
A concrete, quoted, serialized, or semantically embedded prior output is acceptable and
must not lower overall_score merely because it differs textually from the placeholder.

Output ONLY the JSON object. Do not wrap it in code fences or add commentary.

SYSTEM PROMPT:
{system_prompt}

USER REQUEST:
{user_request}

GOLDEN TRUTH:
{golden_truth}

MODEL RESPONSE:
{model_response}
"""


def _build_prompts(
    eval_dataset: EvalDataset,
) -> Dict[str, str]:
    """One judge prompt per conversation, keyed by conversation (scenario) id."""
    test_cases = dataset_to_deepeval_llm_test_cases(eval_dataset)
    system_prompts = {
        convo.id: (convo.metadata or {}).get("system_prompt", "")
        for convo in eval_dataset.conversations
    }

    prompts: Dict[str, str] = {}
    for idx, tc in test_cases.items():
        prompts[idx] = _JUDGE_INSTRUCTIONS.format(
            system_prompt=system_prompts.get(idx, "") or "(system prompt unavailable)",
            user_request=tc.input or "",
            golden_truth=tc.expected_output or "(no reference provided)",
            model_response=tc.actual_output or "(empty response)",
        )
    return prompts


def _build_results(
    name: str,
    eval_dataset: EvalDataset,
    threshold: float,
    verdicts: Dict[str, Any],
    run_id: str,
) -> MetricsResults:
    metrics: Dict[str, _FailureModeResult] = {}
    for convo in eval_dataset.conversations:
        result = _FailureModeResult(name=name, threshold=threshold)
        verdict = verdicts.get(convo.id)
        if isinstance(verdict, ToolCallFailureModes):
            result.apply(verdict)
        else:
            result.apply_parse_error(verdict)
        metrics[convo.id] = result

    return MetricsResults(
        name=name,
        metrics=cast(Any, metrics),
        run_id=run_id,
        dataset_info=eval_dataset.metadata,
    )


# ---------------------------------------------------------------------------
# Staged public API (mirrors accuracy_with_reference)
# ---------------------------------------------------------------------------

def start_tool_call_failure_modes(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    """Enqueue the judge prompts and return immediately (no scores yet)."""
    model = _build_metric_model(evaluator_info, eval_dataset, _METRIC_NAME, environment)
    prompts = _build_prompts(eval_dataset)
    model.request_dict.metric_phase = _PHASE
    return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)


def finalize_tool_call_failure_modes(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    """Poll the enqueued judge batch.

    Returns a status dict (``is_complete=False``) while the batch is pending,
    and a ``MetricsResults`` once every item has been scored.
    """
    model = _build_metric_model(evaluator_info, eval_dataset, _METRIC_NAME, environment)
    prompts = _build_prompts(eval_dataset)

    result = _finalize_phase_or_status(model, prompts, _PHASE, ToolCallFailureModes)
    if isinstance(result, dict) and "is_complete" in result and not result.get("is_complete"):
        return result

    return _build_results(
        name=_METRIC_NAME,
        eval_dataset=eval_dataset,
        threshold=threshold,
        verdicts=cast(Dict[str, Any], result),
        run_id=model.request_dict.run_id,
    )


def tool_call_failure_modes_batch_generate(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
) -> MetricsResults:
    """Single-shot convenience: enqueue, wait for completion, and score."""
    model = _build_metric_model(evaluator_info, eval_dataset, _METRIC_NAME, environment)
    prompts = _build_prompts(eval_dataset)
    model.request_dict.metric_phase = _PHASE

    verdicts = model.batch_generate(prompts=prompts, schema=ToolCallFailureModes)
    return _build_results(
        name=_METRIC_NAME,
        eval_dataset=eval_dataset,
        threshold=threshold,
        verdicts=cast(Dict[str, Any], verdicts),
        run_id=model.request_dict.run_id,
    )
