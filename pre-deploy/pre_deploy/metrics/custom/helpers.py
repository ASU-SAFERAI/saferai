import logging
from typing import Any, Dict, List, Optional, Union

from deepeval.metrics import GEval
from deepeval.metrics.g_eval.schema import ReasonScore
from deepeval.metrics.g_eval.template import GEvalTemplate
from deepeval.metrics.g_eval.utils import (
    Rubric,
    construct_g_eval_params_string,
    construct_test_case_string,
    format_rubrics,
    number_evaluation_steps,
)
from deepeval.metrics.utils import check_llm_test_case_params
from deepeval.test_case import LLMTestCaseParams

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _phase_is_complete as _phase_status

logger = logging.getLogger(__name__)


def _build_evaluation_params(with_reference: bool) -> List[LLMTestCaseParams]:
    evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
    if with_reference:
        evaluation_params.append(LLMTestCaseParams.EXPECTED_OUTPUT)
    return evaluation_params


def _normalize_rubrics(
    rubrics: Union[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], None]
) -> Union[List[Rubric], Dict[str, List[Rubric]], None]:
    if not rubrics:
        return None

    if isinstance(rubrics, dict):
        return {
            idx: [
                Rubric(score_range=rubric["score_range"], expected_outcome=rubric["expected_outcome"])
                for rubric in rubric_list
            ]
            for idx, rubric_list in rubrics.items()
        }

    return [
        Rubric(score_range=rubric["score_range"], expected_outcome=rubric["expected_outcome"])
        for rubric in rubrics
    ]


def _build_metrics(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    name: str,
    evaluation_params: List[LLMTestCaseParams],
    evaluation_steps: Union[List[str], Dict[str, List[str]]],
    rubrics: Union[List[Rubric], Dict[str, List[Rubric]], None],
    threshold: float,
    strict_mode: bool,
) -> Dict[str, GEval]:
    return {
        convo.id: GEval(
            name=name,
            evaluation_params=evaluation_params,
            evaluation_steps=(
                evaluation_steps
                if isinstance(evaluation_steps, list)
                else evaluation_steps.get(convo.id, [])
            ),
            model=model,
            threshold=threshold,
            rubric=(rubrics.get(convo.id, None) if isinstance(rubrics, dict) else rubrics),
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=False,
        )
        for convo in eval_dataset.conversations
    }


def _validate_test_cases(test_cases_dict: Dict[str, Any], metrics: Dict[str, GEval]) -> None:
    for idx, metric in metrics.items():
        check_llm_test_case_params(test_cases_dict[idx], metric.evaluation_params, metric)


def _build_eval_prompts(
    metrics: Dict[str, GEval],
    test_cases_dict: Dict[str, Any],
    strict_mode: bool,
) -> Dict[str, str]:
    test_case_contents = {
        idx: construct_test_case_string(metric.evaluation_params, test_cases_dict[idx])
        for idx, metric in metrics.items()
    }

    g_eval_params_strings = {
        idx: construct_g_eval_params_string(metric.evaluation_params)
        for idx, metric in metrics.items()
    }

    if not strict_mode:
        rubric_strings = {
            idx: format_rubrics(metric.rubric) if metric.rubric else None
            for idx, metric in metrics.items()
        }
        return {
            idx: GEvalTemplate.generate_evaluation_results(
                evaluation_steps=number_evaluation_steps(metric.evaluation_steps),
                test_case_content=test_case_contents[idx],
                parameters=g_eval_params_strings[idx],
                rubric=rubric_strings[idx],
                score_range=metric.score_range,
            )
            for idx, metric in metrics.items()
        }

    return {
        idx: GEvalTemplate.generate_strict_evaluation_results(
            evaluation_steps=number_evaluation_steps(metric.evaluation_steps),
            test_case_content=test_case_contents[idx],
            parameters=g_eval_params_strings[idx],
        )
        for idx, metric in metrics.items()
    }


def _apply_reason_scores(metrics: Dict[str, GEval], eval_responses: Dict[str, Any]) -> None:
    for idx, metric in metrics.items():
        response = eval_responses.get(idx)
        if isinstance(response, ReasonScore):
            metric.score = float(response.score) / 10
            metric.score = (
                0 if metric.strict_mode and metric.score < metric.threshold else metric.score
            )
            metric.reason = response.reason
            metric.success = metric.score >= metric.threshold
        else:
            metric.score = 0
            metric.reason = "error_parsing_response"
            metric.success = False
            logger.error(
                "Error parsing evaluation response: %s. Defaulting to score of 0 and reason of 'error_parsing_response'.",
                response,
            )


def _phase_is_complete(
    model: DeepEvalClient,
    prompts: Dict[str, Any],
    phase: str,
) -> bool:
    return _phase_status(model=model, prompts=prompts, phase=phase)["is_complete"]
