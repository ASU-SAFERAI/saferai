from typing import Any, Dict, List, Optional, Union, cast

from deepeval.metrics.g_eval.schema import ReasonScore

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output import MetricsResults
from ...query_processor import AWSEnvironment, RequestDict
from ..utils import _build_metric_model
from .helpers import (
    _apply_reason_scores,
    _build_eval_prompts,
    _build_evaluation_params,
    _build_metrics,
    _normalize_rubrics,
    _validate_test_cases,
)


def custom_batch_generate(
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

    model.request_dict.metric_phase = "reason_score"
    eval_responses = model.batch_generate(
        prompts=_build_eval_prompts(metrics, test_cases_dict, strict_mode),
        schema=ReasonScore,
    )
    _apply_reason_scores(metrics, cast(Dict[str, Any], eval_responses))

    results = MetricsResults(
        name=name,
        metrics=cast(Any, metrics),
        run_id=model.request_dict.run_id,
        dataset_info=eval_dataset.metadata,
    )

    if not verbose_mode:
        return results
    return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))
