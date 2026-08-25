from typing import Any, Dict, Optional, Union, cast

from deepeval.metrics.conversation_completeness.schema import (
    ConversationCompletenessScoreReason,
    ConversationCompletenessVerdict,
    UserIntentions,
)

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status, _is_incomplete_status
from .helpers import (
    METRIC_NAME,
    _apply_reasons,
    _apply_scores_and_success,
    _apply_user_intentions,
    _apply_verdicts,
    _build_metrics,
    _build_reason_prompts,
    _build_user_intentions_prompts,
    _build_verdicts_prompts,
)


def user_intentions_batch_generate(
    model: DeepEvalClient,
    test_cases_dict: Dict[str, Any],
    enqueue_only: bool = True,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, UserIntentions]]:
    """Run only the user_intentions stage."""
    prompts = _build_user_intentions_prompts(test_cases_dict)
    model.request_dict.metric_phase = "user_intentions"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "user_intentions", UserIntentions)


def verdicts_batch_generate(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    test_cases_dict: Dict[str, Any],
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
    window_size: int = 3,
    enqueue_only: bool = True,
    force_rerun: bool = False,
    user_intentions_responses: Optional[Dict[str, UserIntentions]] = None,
) -> Union[Dict[str, Any], Dict[str, ConversationCompletenessVerdict]]:
    """Run only the verdicts stage."""
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode, window_size)

    if user_intentions_responses is None:
        user_intentions_result = user_intentions_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=False,
        )
        if _is_incomplete_status(user_intentions_result):
            return user_intentions_result
        user_intentions_responses = cast(Dict[str, UserIntentions], user_intentions_result)

    _apply_user_intentions(metrics, cast(Dict[str, Any], user_intentions_responses))

    prompts = _build_verdicts_prompts(test_cases_dict, metrics)
    model.request_dict.metric_phase = "verdicts"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(model, prompts, "verdicts", ConversationCompletenessVerdict)


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
    user_intentions_responses: Optional[Dict[str, UserIntentions]] = None,
    verdicts_responses: Optional[Dict[str, ConversationCompletenessVerdict]] = None,
) -> Union[Dict[str, Any], Dict[str, ConversationCompletenessScoreReason]]:
    """Run only the reasons stage."""
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode, window_size)

    if user_intentions_responses is None:
        user_intentions_result = user_intentions_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=False,
        )
        if _is_incomplete_status(user_intentions_result):
            return user_intentions_result
        user_intentions_responses = cast(Dict[str, UserIntentions], user_intentions_result)

    _apply_user_intentions(metrics, cast(Dict[str, Any], user_intentions_responses))

    if verdicts_responses is None:
        verdicts_result = verdicts_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            test_cases_dict=test_cases_dict,
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            window_size=window_size,
            enqueue_only=False,
            user_intentions_responses=user_intentions_responses,
        )
        if _is_incomplete_status(verdicts_result):
            return verdicts_result
        verdicts_responses = cast(Dict[str, ConversationCompletenessVerdict], verdicts_result)

    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))
    _apply_scores_and_success(metrics)

    prompts = _build_reason_prompts(metrics)
    model.request_dict.metric_phase = "reasons"

    if enqueue_only:
        return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
    return _finalize_phase_or_status(
        model,
        prompts,
        "reasons",
        ConversationCompletenessScoreReason,
    )


def start_conversation_completeness(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    return cast(
        Dict[str, Any],
        user_intentions_batch_generate(
            model=model,
            test_cases_dict=test_cases_dict,
            enqueue_only=True,
            force_rerun=force_rerun,
        ),
    )

def advance_conversation_completeness(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    threshold: float = 0.5,
    window_size: int = 3,
    strict_mode: bool = False,
    include_reason: bool = True,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    if phase not in {"verdicts", "reasons"}:
        raise ValueError("phase must be one of: verdicts, reasons")
    if phase == "reasons" and not include_reason:
        raise ValueError("Cannot advance to reasons when include_reason=False")

    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)

    user_intentions_result = user_intentions_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        enqueue_only=False,
    )
    if _is_incomplete_status(user_intentions_result):
        return cast(Dict[str, Any], user_intentions_result)
    user_intentions_responses = cast(Dict[str, UserIntentions], user_intentions_result)

    if phase == "verdicts":
        return verdicts_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            test_cases_dict=test_cases_dict,
            threshold=threshold,
            include_reason=include_reason,
            strict_mode=strict_mode,
            window_size=window_size,
            enqueue_only=True,
            force_rerun=force_rerun,
            user_intentions_responses=user_intentions_responses,
        )

    verdicts_result = verdicts_batch_generate(
        model=model,
        eval_dataset=eval_dataset,
        test_cases_dict=test_cases_dict,
        threshold=threshold,
        include_reason=include_reason,
        strict_mode=strict_mode,
        window_size=window_size,
        enqueue_only=False,
        user_intentions_responses=user_intentions_responses,
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
        user_intentions_responses=user_intentions_responses,
        verdicts_responses=cast(Dict[str, ConversationCompletenessVerdict], verdicts_result),
    )


def finalize_conversation_completeness(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    window_size: int = 3,
    strict_mode: bool = False,
    include_reason: bool = True,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any], bool]:
    model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
    test_cases_dict = dataset_to_deepeval_conversational_test_cases(eval_dataset)
    metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode, window_size)

    user_intentions_result = user_intentions_batch_generate(
        model=model,
        test_cases_dict=test_cases_dict,
        enqueue_only=False,
    )
    if _is_incomplete_status(user_intentions_result):
        return user_intentions_result
    user_intentions_responses = cast(Dict[str, UserIntentions], user_intentions_result)
    _apply_user_intentions(metrics, cast(Dict[str, Any], user_intentions_responses))

    verdicts_result = verdicts_batch_generate(
        model=model,
        eval_dataset=eval_dataset,
        test_cases_dict=test_cases_dict,
        threshold=threshold,
        include_reason=include_reason,
        strict_mode=strict_mode,
        window_size=window_size,
        enqueue_only=False,
        user_intentions_responses=user_intentions_responses,
    )
    if _is_incomplete_status(verdicts_result):
        return verdicts_result
    verdicts_responses = cast(Dict[str, ConversationCompletenessVerdict], verdicts_result)
    _apply_verdicts(metrics, cast(Dict[str, Any], verdicts_responses))
    _apply_scores_and_success(metrics)

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
            user_intentions_responses=user_intentions_responses,
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
    return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))
