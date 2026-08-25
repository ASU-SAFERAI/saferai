from typing import Any, Dict, List, Optional, Union, cast

from deepeval.metrics import ConversationalGEval
from deepeval.metrics.conversational_g_eval.schema import ReasonScore, Steps

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status, _is_incomplete_status
from .helpers import (
    _apply_evaluation_steps,
    _apply_reason_scores,
    _build_eval_prompts,
    _build_evaluation_steps_prompts,
    _build_metrics,
    _normalize_rubrics,
    _validate_test_cases,
)


def _has_predefined_evaluation_steps(metrics: Dict[str, ConversationalGEval]) -> bool:
    return all(bool(metric.evaluation_steps) for metric in metrics.values())


def evaluation_steps_batch_generate(
    model: DeepEvalClient,
    metrics: Dict[str, ConversationalGEval],
    enqueue_only: bool = True,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, Steps]]:
    """Run only the evaluation_steps stage."""
    prompts = _build_evaluation_steps_prompts(metrics)

    if not prompts:
        if enqueue_only:
            return {
                "run_id": model.request_dict.run_id,
                "metric_phase": "evaluation_steps",
                "total_items": 0,
                "completed_items": 0,
                "pending_items": 0,
                "is_complete": True,
                "query_keys": [],
            }
        return {}

    model.request_dict.metric_phase = "evaluation_steps"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "evaluation_steps", Steps)


def reason_score_batch_generate(
    model: DeepEvalClient,
    test_cases_dict: Dict[str, Any],
    metrics: Dict[str, ConversationalGEval],
    strict_mode: bool = False,
    enqueue_only: bool = True,
    force_rerun: bool = False,
    evaluation_steps_responses: Optional[Dict[str, Steps]] = None,
) -> Union[Dict[str, Any], Dict[str, ReasonScore]]:
    """Run only the reason_score stage."""
    if evaluation_steps_responses is None and not _has_predefined_evaluation_steps(metrics):
        evaluation_steps_result = evaluation_steps_batch_generate(
            model=model,
            metrics=metrics,
            enqueue_only=False,
        )
        if _is_incomplete_status(evaluation_steps_result):
            return evaluation_steps_result
        _apply_evaluation_steps(metrics, cast(Dict[str, Any], evaluation_steps_result))
    elif evaluation_steps_responses is not None:
        _apply_evaluation_steps(metrics, cast(Dict[str, Any], evaluation_steps_responses))

    prompts = _build_eval_prompts(metrics, test_cases_dict, strict_mode)
    model.request_dict.metric_phase = "reason_score"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "reason_score", ReasonScore)


def start_conversation_geval(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    name: str,
    evaluation_steps: List[str],
    rubrics: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if not evaluation_steps:
        raise ValueError("Either criteria or evaluation_steps must be provided.")

    model = _build_metric_model(evaluator_info, eval_dataset, name, environment)
    metrics = _build_metrics(
        model=model,
        eval_dataset=eval_dataset,
        name=name,
        evaluation_steps=evaluation_steps,
        standard_rubrics=_normalize_rubrics(rubrics),
        threshold=threshold,
        strict_mode=strict_mode,
    )
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    _validate_test_cases(test_cases_dict, metrics)

    if _has_predefined_evaluation_steps(metrics):
        return cast(
            Dict[str, Any],
            reason_score_batch_generate(
                model=model,
                test_cases_dict=test_cases_dict,
                metrics=metrics,
                strict_mode=strict_mode,
                enqueue_only=True,
                force_rerun=force_rerun,
            ),
        )

    return cast(
        Dict[str, Any],
        evaluation_steps_batch_generate(
            model=model,
            metrics=metrics,
            enqueue_only=True,
            force_rerun=force_rerun,
        ),
    )

def advance_conversation_geval(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    name: str,
    evaluation_steps: List[str],
    rubrics: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if phase not in {"evaluation_steps", "reason_score"}:
        raise ValueError("phase must be one of: evaluation_steps, reason_score")
    if not evaluation_steps:
        raise ValueError("Either criteria or evaluation_steps must be provided.")

    model = _build_metric_model(evaluator_info, eval_dataset, name, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    metrics = _build_metrics(
        model=model,
        eval_dataset=eval_dataset,
        name=name,
        evaluation_steps=evaluation_steps,
        standard_rubrics=_normalize_rubrics(rubrics),
        threshold=threshold,
        strict_mode=strict_mode,
    )
    _validate_test_cases(test_cases_dict, metrics)

    if _has_predefined_evaluation_steps(metrics):
        if phase == "evaluation_steps":
            return {
                "run_id": model.request_dict.run_id,
                "metric_phase": "evaluation_steps",
                "total_items": 0,
                "completed_items": 0,
                "pending_items": 0,
                "is_complete": True,
                "query_keys": [],
            }

        return cast(
            Dict[str, Any],
            reason_score_batch_generate(
                model=model,
                test_cases_dict=test_cases_dict,
                metrics=metrics,
                strict_mode=strict_mode,
                enqueue_only=True,
                force_rerun=force_rerun,
            ),
        )

    if phase == "evaluation_steps":
        return evaluation_steps_batch_generate(
            model=model,
            metrics=metrics,
            enqueue_only=True,
            force_rerun=force_rerun,
        )

    evaluation_steps_result = evaluation_steps_batch_generate(
        model=model,
        metrics=metrics,
        enqueue_only=False,
    )
    if _is_incomplete_status(evaluation_steps_result):
        return cast(Dict[str, Any], evaluation_steps_result)

    return reason_score_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        metrics=metrics,
        strict_mode=strict_mode,
        enqueue_only=True,
        force_rerun=force_rerun,
        evaluation_steps_responses=cast(Dict[str, Steps], evaluation_steps_result),
    )


def finalize_conversation_geval(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    name: str,
    evaluation_steps: List[str],
    rubrics: Optional[List[Dict[str, Any]]] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    if not evaluation_steps:
        raise ValueError("Either criteria or evaluation_steps must be provided.")

    model = _build_metric_model(evaluator_info, eval_dataset, name, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    metrics = _build_metrics(
        model=model,
        eval_dataset=eval_dataset,
        name=name,
        evaluation_steps=evaluation_steps,
        standard_rubrics=_normalize_rubrics(rubrics),
        threshold=threshold,
        strict_mode=strict_mode,
    )

    _validate_test_cases(test_cases_dict, metrics)

    if _has_predefined_evaluation_steps(metrics):
        reason_score_result = reason_score_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            metrics=metrics,
            strict_mode=strict_mode,
            enqueue_only=False,
        )
        if _is_incomplete_status(reason_score_result):
            return reason_score_result

        _apply_reason_scores(metrics, cast(Dict[str, Any], reason_score_result))

        results = MetricsResults(
            name=name,
            dataset_info=eval_dataset.metadata,
            run_id=model.request_dict.run_id,
            metrics=cast(Any, metrics),
        )

        if not verbose_mode:
            return results
        return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))

    evaluation_steps_result = evaluation_steps_batch_generate(
        model=model,
        metrics=metrics,
        enqueue_only=False,
    )
    if _is_incomplete_status(evaluation_steps_result):
        return evaluation_steps_result
    _apply_evaluation_steps(metrics, cast(Dict[str, Any], evaluation_steps_result))

    reason_score_result = reason_score_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        metrics=metrics,
        strict_mode=strict_mode,
        enqueue_only=False,
        evaluation_steps_responses=cast(Dict[str, Steps], evaluation_steps_result),
    )
    if _is_incomplete_status(reason_score_result):
        return reason_score_result

    _apply_reason_scores(metrics, cast(Dict[str, Any], reason_score_result))

    results = MetricsResults(
        name=name,
        dataset_info=eval_dataset.metadata,
        run_id=model.request_dict.run_id,
        metrics=cast(Any, metrics),
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))
