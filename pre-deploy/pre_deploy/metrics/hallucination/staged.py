from typing import Any, Dict, Optional, Union, cast

from deepeval.metrics.hallucination.schema import HallucinationScoreReason, Verdicts

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status, _is_incomplete_status
from .helpers import (
    METRIC_NAME,
    _apply_reasons,
    _apply_verdicts,
    _build_metrics,
    _build_reasons_prompts,
    _build_verdicts_prompts,
)


def verdicts_batch_generate(
    model: DeepEvalClient,
    test_cases_dict: Dict[str, Any],
    enqueue_only: bool = True,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, Verdicts]]:
    """Run only the verdicts stage."""
    prompts = _build_verdicts_prompts(test_cases_dict)
    model.request_dict.metric_phase = "verdicts"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "verdicts", Verdicts)


def reasons_batch_generate(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    test_cases_dict: Dict[str, Any],
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    enqueue_only: bool = True,
    force_rerun: bool = False,
    verdicts_responses: Optional[Dict[str, Verdicts]] = None,
) -> Union[Dict[str, Any], Dict[str, HallucinationScoreReason]]:
    """Run only the reasons stage."""
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode)

    if verdicts_responses is None:
        verdicts_result = verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=False,
        )
        if _is_incomplete_status(verdicts_result):
            return verdicts_result
        verdicts_responses = cast(Dict[str, Verdicts], verdicts_result)

    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))

    prompts = _build_reasons_prompts(metrics)
    model.request_dict.metric_phase = "reasons"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "reasons", HallucinationScoreReason)


def start_hallucination(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
    return cast(
        Dict[str, Any],
        verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=True,
            force_rerun=force_rerun,
        ),
    )

def advance_hallucination(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if phase not in {"verdicts", "reasons"}:
        raise ValueError("phase must be one of: verdicts, reasons")
    if phase == "reasons" and not include_reason:
        raise ValueError("Cannot advance to reasons when include_reason=False")

    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)

    if phase == "verdicts":
        return verdicts_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=True,
            force_rerun=force_rerun,
        )

    verdicts_result = verdicts_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
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
        enqueue_only=True,
        force_rerun=force_rerun,
        verdicts_responses=cast(Dict[str, Verdicts], verdicts_result),
    )


def finalize_hallucination(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode)

    verdicts_result = verdicts_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        enqueue_only=False,
    )
    if _is_incomplete_status(verdicts_result):
        return verdicts_result
    verdicts_responses = cast(Dict[str, Verdicts], verdicts_result)
    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))

    if include_reason:
        reasons_result = reasons_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            test_cases_dict=test_cases_dict,
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            enqueue_only=False,
            verdicts_responses=verdicts_responses,
        )
        if _is_incomplete_status(reasons_result):
            return reasons_result
        _apply_reasons(metrics, cast(Dict[str, Any], reasons_result))

    results = MetricsResults(
        name=METRIC_NAME,
        dataset_info=eval_dataset.metadata,
        run_id=model.request_dict.run_id,
        metrics=cast(Any, metrics),
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=test_cases_dict)
