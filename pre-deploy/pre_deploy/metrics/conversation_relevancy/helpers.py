import logging
from typing import Any, Dict, List, Optional, cast
import itertools

from deepeval.metrics import TurnRelevancyMetric
from deepeval.metrics.turn_relevancy.schema import TurnRelevancyScoreReason, TurnRelevancyVerdict
from deepeval.metrics.turn_relevancy.template import TurnRelevancyTemplate
from deepeval.metrics.utils import (
    check_conversational_test_case_params,
    convert_turn_to_dict,
    get_turns_in_sliding_window,
    get_unit_interactions,
)

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _extract_idx, _phase_is_complete, _prompt_key

logger = logging.getLogger(__name__)

METRIC_NAME = "conversation_relevancy"


def _build_metrics(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    threshold: float,
    include_reason: bool,
    strict_mode: bool,
    window_size: int,
) -> Dict[str, TurnRelevancyMetric]:
    return {
        convo.id: TurnRelevancyMetric(
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=False,
            window_size=window_size,
        )
        for convo in eval_dataset.conversations
    }


def _validate_test_cases(
    test_cases_dict: Dict[str, Any],
    metrics: Dict[str, TurnRelevancyMetric],
) -> None:
    for idx, metric in metrics.items():
        check_conversational_test_case_params(
            test_cases_dict[idx], TurnRelevancyMetric._required_test_case_params, metric
        )


def _build_turn_windows(
    test_cases_dict: Dict[str, Any],
    window_size: int,
) -> Dict[str, Any]:
    unit_interactions_dict = {
        idx: get_unit_interactions(test_case.turns)
        for idx, test_case in test_cases_dict.items()
    }
    return {
        idx: list(
            itertools.chain(
                *get_turns_in_sliding_window(cast(Any, unit_interactions), window_size=window_size)
            )
        )
        for idx, unit_interactions in unit_interactions_dict.items()
    }


def _build_verdicts_prompts(turns_windows: Dict[str, Any]) -> Dict[str, str]:
    return {
        _prompt_key(idx, i): TurnRelevancyTemplate.generate_verdicts(
            sliding_window=[convert_turn_to_dict(turn) for turn in window]
        )
        for idx, turn_sliding_window in turns_windows.items()
        for i, window in enumerate(turn_sliding_window)
    }


def _apply_verdicts(
    metrics: Dict[str, TurnRelevancyMetric],
    verdicts_responses: Dict[str, Any],
) -> None:
    for metric in metrics.values():
        metric.verdicts = []

    for key, verdict in verdicts_responses.items():
        idx = _extract_idx(key)
        if isinstance(verdict, TurnRelevancyVerdict):
            metrics[idx].verdicts.append(verdict)
        else:
            metrics[idx].verdicts.append(
                TurnRelevancyVerdict(verdict="not_relevant", reason="error_parsing_verdict")
            )


def _apply_scores(metrics: Dict[str, TurnRelevancyMetric]) -> None:
    for metric in metrics.values():
        metric.score = metric._calculate_score()


def _build_reason_prompts(
    metrics: Dict[str, TurnRelevancyMetric],
    eval_dataset: EvalDataset,
) -> Dict[str, str]:
    irrelevancies_dict: Dict[str, List[Dict[str, str]]] = {}
    for idx, metric in metrics.items():
        irrelevancies: List[Dict[str, str]] = []
        for index, verdict in enumerate(metric.verdicts):
            if verdict.verdict.strip().lower() == "no":
                conversations = cast(Any, eval_dataset.conversations)
                message_id = conversations[idx].messages[index].id
                irrelevancies.append(
                    {"message number": str(message_id), "reason": str(verdict.reason)}
                )
        irrelevancies_dict[idx] = irrelevancies

    return {
        idx: TurnRelevancyTemplate.generate_reason(
            score=metric.score,
            irrelevancies=irrelevancies_dict[idx],
        )
        for idx, metric in metrics.items()
    }


def _apply_reasons_and_success(
    metrics: Dict[str, TurnRelevancyMetric],
    reason_responses: Dict[str, Any],
    threshold: float,
) -> None:
    for idx, metric in metrics.items():
        reason = reason_responses.get(idx)
        if isinstance(reason, TurnRelevancyScoreReason):
            metric.reason = reason.reason
        else:
            metric.reason = "reason_not_found"
        metric.success = (metric.score >= threshold) if metric.score is not None else False
