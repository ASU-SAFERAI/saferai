import logging
from typing import Any, Dict, List, Optional

from deepeval.metrics import ConversationalGEval
from deepeval.metrics.conversational_g_eval.schema import ReasonScore, Steps
from deepeval.metrics.conversational_g_eval.template import ConversationalGEvalTemplate
from deepeval.metrics.g_eval.utils import (
    Rubric,
    construct_conversational_g_eval_turn_params_string,
    construct_non_turns_test_case_string,
    format_rubrics,
)
from deepeval.metrics.utils import check_conversational_test_case_params, convert_turn_to_dict

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _phase_is_complete

logger = logging.getLogger(__name__)


def _normalize_rubrics(rubrics: Optional[List[Dict[str, Any]]]) -> Optional[List[Rubric]]:
    if not rubrics:
        return None

    standard_rubrics = []
    for rubric in rubrics:
        standard_rubrics.append(
            Rubric(score_range=rubric["score_range"], expected_outcome=rubric["expected_outcome"])
        )
    return standard_rubrics


def _build_metrics(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    name: str,
    evaluation_steps: List[str],
    standard_rubrics: Optional[List[Rubric]],
    threshold: float,
    strict_mode: bool,
) -> Dict[str, ConversationalGEval]:
    return {
        convo.id: ConversationalGEval(
            name=name,
            evaluation_steps=evaluation_steps,
            criteria=None,
            model=model,
            threshold=threshold,
            rubric=standard_rubrics,
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=False,
        )
        for convo in eval_dataset.conversations
    }


def _validate_test_cases(
    test_cases_dict: Dict[str, Any],
    metrics: Dict[str, ConversationalGEval],
) -> None:
    for idx, metric in metrics.items():
        check_conversational_test_case_params(
            test_cases_dict[idx], metric.evaluation_params, metric
        )


def _build_evaluation_steps_prompts(metrics: Dict[str, ConversationalGEval]) -> Dict[str, str]:
    prompts: Dict[str, str] = {}
    for idx, metric in metrics.items():
        if not metric.evaluation_steps:
            metric.evaluation_steps = []
            g_eval_params_str = construct_conversational_g_eval_turn_params_string(
                metric.evaluation_params
            )
            prompts[idx] = ConversationalGEvalTemplate.generate_evaluation_steps(
                criteria=metric.criteria or "",
                parameters=g_eval_params_str,
            )
    return prompts


def _apply_evaluation_steps(
    metrics: Dict[str, ConversationalGEval],
    evaluation_steps: Dict[str, Any],
) -> None:
    for idx, eval_steps in evaluation_steps.items():
        metric = metrics[idx]
        if isinstance(eval_steps, Steps):
            metric.evaluation_steps = eval_steps.steps
        else:
            metric.evaluation_steps = [metric.criteria or ""]
            metric.success = False
            logger.error(
                "Error parsing evaluation steps: %s. Defaulting to single step with criteria as the step.",
                eval_steps,
            )


def _build_eval_prompts(
    metrics: Dict[str, ConversationalGEval],
    test_cases_dict: Dict[str, Any],
    strict_mode: bool,
) -> Dict[str, str]:
    test_case_contents = {
        idx: construct_non_turns_test_case_string(metric.evaluation_params, test_cases_dict[idx])
        for idx, metric in metrics.items()
    }

    g_eval_params_strings = {
        idx: construct_conversational_g_eval_turn_params_string(metric.evaluation_params)
        for idx, metric in metrics.items()
    }

    if not strict_mode:
        rubric_strings = {
            idx: format_rubrics(metric.rubric) if metric.rubric else None
            for idx, metric in metrics.items()
        }
        return {
            idx: ConversationalGEvalTemplate.generate_evaluation_results(
                evaluation_steps=metric.number_evaluation_steps(),
                test_case_content=test_case_contents[idx],
                turns=[
                    convert_turn_to_dict(turn, metric.evaluation_params)
                    for turn in test_cases_dict[idx].turns
                ],
                parameters=g_eval_params_strings[idx],
                rubric=rubric_strings[idx],
            )
            for idx, metric in metrics.items()
        }

    return {
        idx: ConversationalGEvalTemplate.generate_evaluation_results(
            evaluation_steps=metric.number_evaluation_steps(),
            test_case_content=test_case_contents[idx],
            turns=[
                convert_turn_to_dict(turn, metric.evaluation_params)
                for turn in test_cases_dict[idx].turns
            ],
            parameters=g_eval_params_strings[idx],
            rubric=None,
        )
        for idx, metric in metrics.items()
    }


def _apply_reason_scores(
    metrics: Dict[str, ConversationalGEval],
    eval_responses: Dict[str, Any],
) -> None:
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
