"""Reporting: per-scenario detail rows and denominator-explicit summaries.

Owns the analytical output of a run. Turning finished frames into files on disk
(provenance, manifest, CSV/XLSX) lives in :mod:`artifacts`.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from pre_deploy.input import EvalDataset
from pre_deploy.metrics.tool_call_failure_modes import DIMENSION_KEYS
from pre_deploy.output import MetricsResults
from custom_benchmarks.tool_calling.serialization import (
    canonical_json,
    serialize_audit_observed_trace,
    serialize_expected_trace,
    serialize_observed_trace,
)

from .config import THRESHOLD
from .data import _EXECUTION_FAILURE_STATUSES, _scenario_metadata


# ---------------------------------------------------------------------------
# Audit view
# ---------------------------------------------------------------------------

# The pared-down column set written to the detail CSV/XLSX. It is scoped to what
# an evaluator auditing a run needs: candidate/scenario identity, dataset
# metadata (including the ``meta_*`` scenario descriptors), the golden-truth
# trace, the tool calls the candidate actually produced, and the judge verdict
# with its per-dimension flags. Heavy forensic/transport columns (request,
# stream_summary, eos_metadata, raw errors, duplicate/system-prompt fields) are
# intentionally excluded. ``meta_*`` scenario-metadata columns are appended
# dynamically at projection time because their exact set depends on the dataset.
AUDIT_COLUMNS: list[str] = [
    # Identity
    "run_id",
    "model",
    "provider",
    "scenario_id",
    "combination_id",
    "variant",
    # Dataset metadata / request
    "question",
    "expected_tools",
    "enabled_tools",
    "hop_count",
    # Ground truth
    "expected_calls",
    # Candidate output (tool_name + parsed arguments only; full tool outputs
    # and provider metadata live in the opt-in raw observations JSON).
    "observed_calls",
    "final_response",
    # Verdict
    "score",
    "success",
    "reason",
    "evaluation_status",
    "metric_name",
    "threshold",
    # Per-dimension judge flags
    *DIMENSION_KEYS,
]


def audit_columns_for(columns: Any) -> list[str]:
    """Return the audit column order restricted to columns present in a frame.

    Any ``meta_*`` scenario-metadata columns present in the detail frame are
    appended (sorted) after the fixed audit columns so dataset-specific
    descriptors survive the projection without being enumerated up front.
    """
    available = list(columns)
    available_set = set(available)
    ordered = [column for column in AUDIT_COLUMNS if column in available_set]
    ordered_set = set(ordered)
    meta_columns = sorted(
        column
        for column in available
        if column not in ordered_set and str(column).startswith("meta_")
    )
    return ordered + meta_columns


# ---------------------------------------------------------------------------
# Conversation and evaluator context helpers
# ---------------------------------------------------------------------------

def _message_text(conv, sequence_index: int) -> str:
    """Safely extract the text content of a conversation message by index.

    Mirrors how dataset_to_deepeval_llm_test_cases reads input (index 0) and
    actual_output (index 1): messages[i].contents[0].content.
    """
    if conv is None:
        return ""
    try:
        return conv.messages[sequence_index].contents[0].content or ""
    except (IndexError, AttributeError):
        return ""


def _conversation_input(conv) -> str:
    """The benchmark question (user message)."""
    return _message_text(conv, 0)


def _conversation_actual_output(conv) -> str:
    """The candidate model's response (assistant message)."""
    return _message_text(conv, 1)


def _report_evaluator_context(
    scored: MetricsResults | None,
    eval_dataset: EvalDataset | None,
    evaluator_model: Mapping[str, Any] | None,
    evaluator_status: str | None,
    evaluator_failure: str | None,
) -> dict[str, Any]:
    """Resolve evaluator fields from the lifecycle result and dataset metadata."""
    dataset_info = {}
    if eval_dataset is not None:
        dataset_info.update(eval_dataset.metadata)
    if scored is not None and isinstance(scored.dataset_info, Mapping):
        dataset_info.update(scored.dataset_info)

    selected_model = evaluator_model or {}
    context = {
        "evaluator_model": selected_model.get("name") or dataset_info.get("evaluator_model"),
        "evaluator_provider": selected_model.get("provider") or dataset_info.get("evaluator_provider"),
        "evaluator_run_id": dataset_info.get("evaluator_run_id")
        or (scored.run_id if scored is not None else None),
        "metric": dataset_info.get("metric")
        or dataset_info.get("metric_name")
        or (scored.name if scored is not None else "tool_call_failure_modes"),
        "metric_name": dataset_info.get("metric_name")
        or dataset_info.get("metric")
        or (scored.name if scored is not None else "tool_call_failure_modes"),
        "threshold": dataset_info.get("threshold", THRESHOLD),
        "evaluator_status": evaluator_status
        or dataset_info.get("evaluator_status")
        or ("completed" if scored is not None else "not_run"),
        "evaluator_failure": evaluator_failure
        or dataset_info.get("evaluator_failure"),
    }
    return context


def _scenario_records(
    scenarios: list[dict] | None,
    eval_dataset: EvalDataset | None,
) -> list[tuple[str, dict[str, Any], Any]]:
    """Return source-order scenario IDs, source records, and eval conversations."""
    conversations = {
        conversation.id: conversation
        for conversation in (eval_dataset.conversations if eval_dataset is not None else [])
    }
    if scenarios is not None:
        return [
            (scenario["scenario_id"], scenario, conversations.get(scenario["scenario_id"]))
            for scenario in scenarios
        ]
    return [
        (scenario_id, {}, conversations[scenario_id])
        for scenario_id in conversations
    ]


# ---------------------------------------------------------------------------
# Detail report
# ---------------------------------------------------------------------------

def build_report(
    model_name: str,
    scored: MetricsResults | None,
    eval_dataset: EvalDataset | None,
    *,
    model_provider: str | None = None,
    observations: Mapping[str, ToolCallObservation] | None = None,
    scenarios: list[dict] | None = None,
    evaluator_model: Mapping[str, Any] | None = None,
    evaluator_status: str | None = None,
    evaluator_failure: str | None = None,
) -> list[dict]:
    """Build one report row per attempted scenario, including failures.

    Evaluator results are keyed by scenario ID rather than positional order.
    Execution failures are assigned score ``0`` and
    ``not_scored_execution_failure``; evaluator failures retain the candidate
    observation and use a null score with ``evaluation_failed``.  This keeps a
    failed or excluded item visible without submitting an empty trace to the
    judge as a successful abstention.
    """
    scored_payload = scored.to_dict() if scored is not None else {}
    scores = scored_payload.get("results", {})
    observations = observations or {}
    evaluator = _report_evaluator_context(
        scored,
        eval_dataset,
        evaluator_model,
        evaluator_status,
        evaluator_failure,
    )

    leading = [
        "run_id", "model", "provider", "scenario_id", "combination_id", "variant",
        "question", "input", "expected_tools", "enabled_tools", "tool_catalog", "hop_count",
        "scenario_metadata", "expected_calls", "observed_calls", "tool_outputs",
        "final_response", "thinking", "request", "stream_summary", "eos_metadata",
        "trace_status", "status", "status_reason", "completed", "eos_received",
        "transport_error", "errors", "expected_call_count", "observed_call_count",
        "argument_parse_errors", "tool_sequence_match", "score", "success", "reason",
        "evaluation_status", "evaluator_status", "evaluator_failure", "evaluator_model",
        "evaluator_provider", "evaluator_run_id", "metric", "metric_name", "threshold",
        "placeholder_propagation", *DIMENSION_KEYS,
    ]

    rows: list[dict] = []
    for scenario_id, scenario, conversation in _scenario_records(scenarios, eval_dataset):
        metadata = dict(conversation.metadata) if conversation is not None else {}
        observation = observations.get(scenario_id)
        result = scores.get(scenario_id)
        dimensions = (result or {}).get("dimensions") or {}

        if observation is not None:
            candidate_run_id = observation.run_id
            candidate_model = observation.model
            candidate_provider = observation.provider
            # ``observed_calls`` is the human-facing audit value: only tool
            # name and parsed arguments. The full trace (with ordinals, tool
            # outputs, and provider metadata) still reaches the evaluator via
            # the eval dataset built in ``data.py`` and is preserved in the
            # opt-in raw observations JSON.
            observed_calls = serialize_audit_observed_trace(observation)
            tool_outputs = canonical_json(
                [call.tool_output for call in observation.tool_calls]
            )
            trace_status = observation.status
            status_reason = observation.status_reason
            completed = observation.completed
            eos_received = observation.eos_received
            transport_error = observation.transport_error
            errors = canonical_json(observation.errors)
            final_response = observation.final_response
            thinking = observation.thinking
            request = canonical_json(observation.request)
            stream_summary = canonical_json(observation.stream_summary)
            eos_metadata = canonical_json(observation.eos_metadata)
            observed_call_count = len(observation.tool_calls)
            argument_parse_errors = canonical_json([
                {
                    "ordinal": call.ordinal,
                    "tool_name": call.tool_name,
                    "error": call.argument_parse_error,
                }
                for call in observation.tool_calls
                if call.argument_parse_error is not None
            ])
            observed_tool_names = [call.tool_name for call in observation.tool_calls]
        else:
            candidate_run_id = None
            candidate_model = model_name
            candidate_provider = model_provider
            observed_calls = None
            tool_outputs = canonical_json([])
            trace_status = "missing_observation"
            status_reason = "no candidate observation was provided"
            completed = False
            eos_received = False
            transport_error = None
            errors = canonical_json([])
            final_response = ""
            thinking = ""
            request = canonical_json({})
            stream_summary = canonical_json({})
            eos_metadata = canonical_json(None)
            observed_call_count = 0
            argument_parse_errors = canonical_json([])
            observed_tool_names = []

        expected_calls = (
            serialize_expected_trace(scenario)
            if scenario
            else metadata.get("expected_output")
        )
        actual_output = (
            _conversation_actual_output(conversation)
            if conversation is not None
            else observed_calls or ""
        )
        scenario_fields = _scenario_metadata(scenario) if scenario else {}
        question = _conversation_input(conversation) or (
            scenario.get("questions", [""])[0] if scenario else ""
        )
        expected_tool_names = list(scenario.get("expected_tools", [])) if scenario else []
        expected_call_count = (
            scenario.get("hop_count", len(expected_tool_names)) if scenario else len(expected_tool_names)
        )
        tool_sequence_match = (
            observed_tool_names == expected_tool_names
            if observation is not None and scenario
            else None
        )

        execution_failed = observation is None or not observation.completed or (
            observation.status in _EXECUTION_FAILURE_STATUSES if observation is not None else True
        )
        if execution_failed:
            score = 0.0
            success = False
            evaluation_status = "not_scored_execution_failure"
            reason = status_reason
            item_evaluator_status = "not_submitted"
            item_evaluator_failure = None
        elif result is None:
            score = None
            success = False
            evaluation_status = "evaluation_failed"
            reason = evaluator.get("evaluator_failure") or "evaluator returned no result for scenario"
            item_evaluator_status = evaluator.get("evaluator_status") or "failed"
            item_evaluator_failure = reason
        else:
            score = result.get("score")
            success = result.get("success")
            reason = result.get("reason")
            item_evaluator_status = evaluator.get("evaluator_status") or "completed"
            if reason == "error_parsing_response":
                evaluation_status = "evaluation_failed"
                item_evaluator_failure = reason
            else:
                evaluation_status = "scored"
                item_evaluator_failure = evaluator.get("evaluator_failure")

        row = {
            "run_id": candidate_run_id,
            "model": candidate_model,
            "provider": candidate_provider,
            "scenario_id": scenario_id,
            "combination_id": scenario.get("combination_id") if scenario else metadata.get("combination_id"),
            "variant": scenario.get("variant") if scenario else metadata.get("variant"),
            "question": question,
            "input": question,
            "model_response": actual_output,
            "expected_tools": canonical_json(expected_tool_names),
            "enabled_tools": canonical_json(scenario.get("enabled_tools", {})) if scenario else metadata.get("enabled_tools"),
            "tool_catalog": scenario_fields.get("tool_catalog") or metadata.get("tool_catalog"),
            "hop_count": expected_call_count,
            "scenario_metadata": scenario_fields.get("scenario_metadata") or metadata.get("scenario_metadata"),
            "expected_calls": expected_calls,
            "observed_calls": observed_calls,
            "tool_outputs": tool_outputs,
            "final_response": final_response,
            "thinking": thinking,
            "request": request,
            "stream_summary": stream_summary,
            "eos_metadata": eos_metadata,
            "trace_status": trace_status,
            "status": trace_status,
            "status_reason": status_reason,
            "completed": completed,
            "eos_received": eos_received,
            "transport_error": transport_error,
            "errors": errors,
            "expected_call_count": expected_call_count,
            "observed_call_count": observed_call_count,
            "argument_parse_errors": argument_parse_errors,
            "tool_sequence_match": tool_sequence_match,
            "score": score,
            "success": success,
            "reason": reason,
            "evaluation_status": evaluation_status,
            "evaluator_status": item_evaluator_status,
            "evaluator_failure": item_evaluator_failure,
            "evaluator_model": evaluator.get("evaluator_model"),
            "evaluator_provider": evaluator.get("evaluator_provider"),
            "evaluator_run_id": evaluator.get("evaluator_run_id"),
            "metric": evaluator.get("metric"),
            "metric_name": evaluator.get("metric_name"),
            "threshold": evaluator.get("threshold"),
            "placeholder_propagation": dimensions.get("placeholder_propagation"),
        }
        for dim in DIMENSION_KEYS:
            row[dim] = dimensions.get(dim)
        for key, value in metadata.items():
            row.setdefault(key, value)
        if scenario:
            for key, value in _scenario_metadata(scenario).items():
                row.setdefault(key, value)

        ordered = {key: row.get(key) for key in leading}
        for key in sorted(row):
            if key not in ordered and key != "reason":
                ordered[key] = row[key]
        ordered["reason"] = reason
        rows.append(ordered)
    return rows


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def build_summary(report_df: pd.DataFrame) -> pd.DataFrame:
    """Build one denominator-explicit summary row per model/provider/run.

    ``questions`` counts every detail row because every row represents an
    attempted scenario.  ``scored_questions`` counts non-null evaluator
    scores, including a valid score of ``0``.  Mean score is calculated only
    over those scored rows, while pass rate uses all attempted rows so
    execution and evaluation failures remain visible as non-passing results.
    Candidate and evaluator identity fields are retained instead of grouping
    models by name alone.
    """
    identity_columns = ["model", "provider", "run_id"]
    optional_identity_columns = [
        "evaluator_model",
        "evaluator_provider",
        "evaluator_run_id",
        "metric",
        "metric_name",
        "threshold",
    ]
    required_columns = {"model", "scenario_id"}
    missing = sorted(required_columns.difference(report_df.columns))
    if missing:
        raise ValueError(
            "report data is missing required summary columns: "
            + ", ".join(missing)
        )

    work = report_df.copy()
    # Detail rows produced by build_report contain these columns.  Supplying
    # nulls for older/custom detail frames keeps the aggregation compatible
    # while making the summary schema explicit.
    for column in identity_columns + optional_identity_columns:
        if column not in work.columns:
            work[column] = None

    dimension_columns = [dim for dim in DIMENSION_KEYS if dim in work.columns]
    summary_columns = [
        "model",
        "provider",
        "run_id",
        "evaluator_model",
        "evaluator_provider",
        "evaluator_run_id",
        "metric",
        "metric_name",
        "threshold",
        "questions",
        "scored_questions",
        "completed",
        "execution_failures",
        "evaluation_failures",
        "mean_score",
        "pass_rate",
        *(f"{dim}_rate" for dim in dimension_columns),
    ]
    if work.empty:
        return pd.DataFrame(columns=summary_columns)

    rows: list[dict[str, Any]] = []
    for identity, group in work.groupby(identity_columns, dropna=False, sort=False):
        questions = len(group)
        scores = pd.to_numeric(group["score"], errors="coerce")
        scored_mask = group["score"].notna()
        successes = group["success"].eq(True).sum() if "success" in group else 0
        row = {
            "model": identity[0],
            "provider": identity[1],
            "run_id": identity[2],
            "evaluator_model": _first_non_null(group["evaluator_model"]),
            "evaluator_provider": _first_non_null(group["evaluator_provider"]),
            "evaluator_run_id": _first_non_null(group["evaluator_run_id"]),
            "metric": _first_non_null(group["metric"]),
            "metric_name": _first_non_null(group["metric_name"]),
            "threshold": _first_non_null(group["threshold"]),
            "questions": questions,
            "scored_questions": int(scored_mask.sum()),
            "completed": int(group["completed"].eq(True).sum()) if "completed" in group else 0,
            "execution_failures": int(
                group["evaluation_status"].eq("not_scored_execution_failure").sum()
            ) if "evaluation_status" in group else 0,
            "evaluation_failures": int(
                group["evaluation_status"].eq("evaluation_failed").sum()
            ) if "evaluation_status" in group else 0,
            "mean_score": scores[scored_mask].mean() if scored_mask.any() else float("nan"),
            # The denominator is attempted scenarios, not scored scenarios.
            "pass_rate": float(successes) / questions if questions else 0.0,
        }
        for dim in dimension_columns:
            row[f"{dim}_rate"] = pd.to_numeric(
                group[dim], errors="coerce"
            ).mean()
        rows.append(row)

    return (
        pd.DataFrame(rows, columns=summary_columns)
        .sort_values(
            ["mean_score", "model", "provider", "run_id"],
            ascending=[False, True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
    )


def _first_non_null(values: pd.Series) -> Any:
    """Return the first non-null value from a grouped detail column."""
    non_null = values.dropna()
    return non_null.iloc[0] if not non_null.empty else None
