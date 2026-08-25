from typing import Any, Dict, List, Optional, Union, cast

from deepeval.metrics import GEval
from deepeval.metrics.g_eval.schema import ReasonScore

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status
from .helpers import (
    _apply_reason_scores,
    _build_eval_prompts,
    _build_evaluation_params,
    _build_metrics,
    _normalize_rubrics,
    _validate_test_cases,
)


def reason_score_batch_generate(
    model: DeepEvalClient,
    test_cases_dict: Dict[str, Any],
    metrics: Dict[str, GEval],
    strict_mode: bool = False,
    enqueue_only: bool = True,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, ReasonScore]]:
    """Run only the reason_score stage."""
    prompts = _build_eval_prompts(metrics, test_cases_dict, strict_mode)
    model.request_dict.metric_phase = "reason_score"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)

    return _finalize_phase_or_status(model, prompts, "reason_score", ReasonScore)


def start_custom(
    name: str,
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    with_reference: bool,
    evaluation_steps: Union[List[str], Dict[str, List[str]]],
    rubrics: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], None] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if not evaluation_steps:
        raise ValueError("evaluation_steps must be provided and cannot be empty.")

    model = _build_metric_model(evaluator_info, eval_dataset, name, environment)
    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
    metrics = _build_metrics(
        model=model,
        eval_dataset=eval_dataset,
        name=name,
        evaluation_params=_build_evaluation_params(with_reference),
        evaluation_steps=evaluation_steps,
        rubrics=_normalize_rubrics(rubrics),
        threshold=threshold,
        strict_mode=strict_mode,
    )

    _validate_test_cases(test_cases_dict, metrics)

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


def advance_custom(
    name: str,
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    with_reference: bool,
    evaluation_steps: Union[List[str], Dict[str, List[str]]],
    rubrics: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], None] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if phase != "reason_score":
        raise ValueError("phase must be: reason_score")

    return start_custom(
        name=name,
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        with_reference=with_reference,
        evaluation_steps=evaluation_steps,
        rubrics=rubrics,
        threshold=threshold,
        strict_mode=strict_mode,
        environment=environment,
        force_rerun=force_rerun,
    )


def finalize_custom(
    name: str,
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    with_reference: bool,
    evaluation_steps: Union[List[str], Dict[str, List[str]]],
    rubrics: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], None] = None,
    threshold: float = 0.5,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    if not evaluation_steps:
        raise ValueError("evaluation_steps must be provided and cannot be empty.")

    model = _build_metric_model(evaluator_info, eval_dataset, name, environment)
    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
    metrics = _build_metrics(
        model=model,
        eval_dataset=eval_dataset,
        name=name,
        evaluation_params=_build_evaluation_params(with_reference),
        evaluation_steps=evaluation_steps,
        rubrics=_normalize_rubrics(rubrics),
        threshold=threshold,
        strict_mode=strict_mode,
    )

    _validate_test_cases(test_cases_dict, metrics)

    reason_score_result = reason_score_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        metrics=metrics,
        strict_mode=strict_mode,
        enqueue_only=False,
    )
    if isinstance(reason_score_result, dict) and "is_complete" in reason_score_result and not reason_score_result.get("is_complete"):
        return reason_score_result

    _apply_reason_scores(metrics, cast(Dict[str, Any], reason_score_result))

    results = MetricsResults(
        name=name,
        metrics=cast(Any, metrics),
        run_id=model.request_dict.run_id,
        dataset_info=eval_dataset.metadata,
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=test_cases_dict)
