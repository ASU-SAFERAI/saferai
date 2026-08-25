from typing import Any, Dict, Optional, Union, cast

from deepeval.metrics.turn_relevancy.schema import TurnRelevancyScoreReason, TurnRelevancyVerdict

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status, _is_incomplete_status
from .helpers import (
    METRIC_NAME,
    _apply_reasons_and_success,
    _apply_scores,
    _apply_verdicts,
    _build_metrics,
    _build_reason_prompts,
    _build_turn_windows,
    _build_verdicts_prompts,
    _validate_test_cases,
)


def verdicts_batch_generate(
    model: DeepEvalClient,
    test_cases_dict: Dict[str, Any],
    window_size: int = 3,
    enqueue_only: bool = True,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, TurnRelevancyVerdict]]:
    """Run only the verdicts stage."""
    prompts = _build_verdicts_prompts(_build_turn_windows(test_cases_dict, window_size))
    model.request_dict.metric_phase = "verdicts"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "verdicts", TurnRelevancyVerdict)


def reasons_batch_generate(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    test_cases_dict: Dict[str, Any],
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    window_size: int = 3,
    enqueue_only: bool = True,
    force_rerun: bool = False,
    verdicts_responses: Optional[Dict[str, TurnRelevancyVerdict]] = None,
) -> Union[Dict[str, Any], Dict[str, TurnRelevancyScoreReason]]:
    """Run only the reasons stage."""
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode, window_size)

    if verdicts_responses is None:
        verdicts_result = verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            window_size=window_size,
            enqueue_only=False,
        )
        if _is_incomplete_status(verdicts_result):
            return verdicts_result
        verdicts_responses = cast(Dict[str, TurnRelevancyVerdict], verdicts_result)

    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))
    _apply_scores(metrics)

    prompts = _build_reason_prompts(metrics, eval_dataset)
    model.request_dict.metric_phase = "reasons"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "reasons", TurnRelevancyScoreReason)


def start_conversation_relevancy(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    window_size: int = 3,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    return cast(
        Dict[str, Any],
        verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            window_size=window_size,
            enqueue_only=True,
            force_rerun=force_rerun,
        ),
    )

def advance_conversation_relevancy(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    window_size: int = 3,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if phase not in {"verdicts", "reasons"}:
        raise ValueError("phase must be one of: verdicts, reasons")
    if phase == "reasons" and not include_reason:
        raise ValueError("Cannot advance to reasons when include_reason=False")

    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)

    if phase == "verdicts":
        return verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            window_size=window_size,
            enqueue_only=True,
            force_rerun=force_rerun,
        )

    verdicts_result = verdicts_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        window_size=window_size,
        enqueue_only=False,
    )
    if _is_incomplete_status(verdicts_result):
        return cast(Dict[str, Any], verdicts_result)

    return reasons_batch_generate(
        model=model,
        eval_dataset=eval_dataset,
        test_cases_dict=test_cases_dict,
        threshold=threshold,
        include_reason=include_reason,
        strict_mode=strict_mode,
        window_size=window_size,
        enqueue_only=True,
        force_rerun=force_rerun,
        verdicts_responses=cast(Dict[str, TurnRelevancyVerdict], verdicts_result),
    )


def finalize_conversation_relevancy(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    window_size: int = 3,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any], bool]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode, window_size)

    _validate_test_cases(test_cases_dict, metrics)

    verdicts_result = verdicts_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        window_size=window_size,
        enqueue_only=False,
    )
    if _is_incomplete_status(verdicts_result):
        return verdicts_result
    verdicts_responses = cast(Dict[str, TurnRelevancyVerdict], verdicts_result)
    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))
    _apply_scores(metrics)

    if include_reason:
        reasons_result = reasons_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            test_cases_dict=test_cases_dict,
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            window_size=window_size,
            enqueue_only=False,
            verdicts_responses=verdicts_responses,
        )
        if _is_incomplete_status(reasons_result):
            return reasons_result
        _apply_reasons_and_success(metrics, cast(Dict[str, Any], reasons_result), threshold)

    results = MetricsResults(
        name=METRIC_NAME,
        dataset_info=eval_dataset.metadata,
        run_id=model.request_dict.run_id,
        metrics=cast(Any, metrics),
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=test_cases_dict)
