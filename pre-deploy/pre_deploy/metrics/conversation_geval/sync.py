from typing import Any, Dict, List, Optional, Union, cast

from deepeval.metrics.conversational_g_eval.schema import ReasonScore, Steps

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, RequestDict
from ..utils import _build_metric_model
from .helpers import (
    _apply_evaluation_steps,
    _apply_reason_scores,
    _build_eval_prompts,
    _build_evaluation_steps_prompts,
    _build_metrics,
    _normalize_rubrics,
    _validate_test_cases,
)


def conversation_geval_batch_generate(
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

    evaluation_step_prompts = _build_evaluation_steps_prompts(metrics)
    if evaluation_step_prompts:
        model.request_dict.metric_phase = "evaluation_steps"
        evaluation_steps_response = model.batch_generate(
            prompts=evaluation_step_prompts,
            schema=Steps,
        )
        _apply_evaluation_steps(metrics, cast(Dict[str, Any], evaluation_steps_response))

    model.request_dict.metric_phase = "reason_score"
    eval_responses = model.batch_generate(
        prompts=_build_eval_prompts(metrics, test_cases_dict, strict_mode),
        schema=ReasonScore,
    )
    _apply_reason_scores(metrics, cast(Dict[str, Any], eval_responses))

    results = MetricsResults(
        name=name,
        dataset_info=eval_dataset.metadata,
        run_id=model.request_dict.run_id,
        metrics=cast(Any, metrics),
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))
