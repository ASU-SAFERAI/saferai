import logging
from collections import defaultdict
from typing import Any, Dict, Optional

from deepeval.metrics import ConversationCompletenessMetric
from deepeval.metrics.conversation_completeness.schema import (
    ConversationCompletenessScoreReason,
    ConversationCompletenessVerdict,
    UserIntentions,
)
from deepeval.metrics.conversation_completeness.template import ConversationCompletenessTemplate
from deepeval.metrics.utils import convert_turn_to_dict

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_conversational_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _extract_idx, _phase_is_complete, _prompt_key

logger = logging.getLogger(__name__)

METRIC_NAME = "conversation_completeness"


def _build_metrics(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    threshold: float,
    include_reason: bool,
    strict_mode: bool,
    window_size: int,
) -> Dict[str, ConversationCompletenessMetric]:
    return {
        convo.id: ConversationCompletenessMetric(
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


def _build_user_intentions_prompts(test_cases_dict: Dict[str, Any]) -> Dict[str, str]:
    return {
        idx: ConversationCompletenessTemplate.extract_user_intentions(
            turns=[convert_turn_to_dict(turn) for turn in test_case.turns]
        )
        for idx, test_case in test_cases_dict.items()
    }


def _apply_user_intentions(
    metrics: Dict[str, ConversationCompletenessMetric],
    user_intentions_responses: Dict[str, Any],
) -> None:
    for idx, metric in metrics.items():
        user_intention_response = user_intentions_responses.get(idx)
        if isinstance(user_intention_response, UserIntentions):
            metric.user_intentions = user_intention_response.intentions
        else:
            logger.error("Error parsing user intentions: %s", user_intention_response)
            metric.user_intentions = UserIntentions(intentions=["error_parsing_intentions"]).intentions


def _build_verdicts_prompts(
    test_cases_dict: Dict[str, Any],
    metrics: Dict[str, ConversationCompletenessMetric],
) -> Dict[str, str]:
    return {
        _prompt_key(idx, i): ConversationCompletenessTemplate.generate_verdicts(
            turns=[convert_turn_to_dict(turn) for turn in test_cases_dict[idx].turns],
            intention=user_intention,
        )
        for idx, metric in metrics.items()
        for i, user_intention in enumerate(metric.user_intentions)
    }


def _apply_verdicts(
    metrics: Dict[str, ConversationCompletenessMetric],
    verdicts_responses: Dict[str, Any],
) -> None:
    def _verdict_parsing(verdict: Any) -> ConversationCompletenessVerdict:
        if isinstance(verdict, ConversationCompletenessVerdict):
            return verdict
        return ConversationCompletenessVerdict(verdict="no", reason="error_parsing_verdict")

    for metric in metrics.values():
        metric.verdicts = []

    for key, verdict in verdicts_responses.items():
        idx = _extract_idx(key)
        metrics[idx].verdicts.append(_verdict_parsing(verdict))


def _apply_scores_and_success(metrics: Dict[str, ConversationCompletenessMetric]) -> None:
    for metric in metrics.values():
        metric.score = metric._calculate_score()
        metric.success = (metric.score >= metric.threshold) if metric.score is not None else False


def _build_reason_prompts(metrics: Dict[str, ConversationCompletenessMetric]) -> Dict[str, str]:
    incompletenesses_dict: Dict[str, Any] = defaultdict(list)
    for idx, metric in metrics.items():
        for verdict in metric.verdicts:
            if verdict.verdict.strip().lower() == "no":
                incompletenesses_dict[idx].append(verdict.verdict.strip())

    return {
        idx: ConversationCompletenessTemplate.generate_reason(
            score=metric.score,
            incompletenesses=incompletenesses_dict[idx],
            intentions=metric.user_intentions,
        )
        for idx, metric in metrics.items()
    }


def _apply_reasons(
    metrics: Dict[str, ConversationCompletenessMetric],
    reason_responses: Dict[str, Any],
) -> None:
    for idx, metric in metrics.items():
        reason = reason_responses.get(idx)
        if isinstance(reason, ConversationCompletenessScoreReason):
            metric.reason = reason.reason
        else:
            metric.reason = "reason_not_found"
