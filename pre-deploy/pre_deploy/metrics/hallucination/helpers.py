import logging
from typing import Any, Dict, Optional

from deepeval.metrics import HallucinationMetric
from deepeval.metrics.hallucination.schema import HallucinationScoreReason, Verdicts
from deepeval.metrics.hallucination.template import HallucinationTemplate

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _phase_is_complete

logger = logging.getLogger(__name__)

METRIC_NAME = "hallucination"


def _build_metrics(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    threshold: float,
    include_reason: bool,
    strict_mode: bool,
) -> Dict[str, HallucinationMetric]:
    return {
        convo.id: HallucinationMetric(
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=False,
        )
        for convo in eval_dataset.conversations
    }


def _build_verdicts_prompts(test_cases_dict: Dict[str, Any]) -> Dict[str, str]:
    return {
        idx: HallucinationTemplate.generate_verdicts(
            actual_output=test_case.actual_output,
            contexts=test_case.context,
        )
        for idx, test_case in test_cases_dict.items()
    }


def _apply_verdicts(
    metrics: Dict[str, HallucinationMetric],
    verdicts_responses: Dict[str, Any],
) -> None:
    for idx, metric in metrics.items():
        verdicts = verdicts_responses.get(idx)
        if isinstance(verdicts, Verdicts):
            metric.verdicts = [item for item in verdicts.verdicts]
            metric.score = metric._calculate_score()
            metric.success = (metric.score >= metric.threshold) if metric.score else False


def _build_reasons_prompts(metrics: Dict[str, HallucinationMetric]) -> Dict[str, Any]:
    def _reason_prompt(metric: HallucinationMetric) -> Any:
        factual_alignments = []
        contradictions = []
        for verdict in metric.verdicts:
            if verdict.verdict.strip().lower() == "yes":
                factual_alignments.append(verdict.reason)
            else:
                contradictions.append(verdict.reason)

        return metric.evaluation_template.generate_reason(
            factual_alignments=factual_alignments,
            contradictions=contradictions,
            score=round(metric.score, 2) if metric.score is not None else 0.0,
        )

    return {idx: _reason_prompt(metric) for idx, metric in metrics.items()}


def _apply_reasons(
    metrics: Dict[str, HallucinationMetric],
    reasons_responses: Dict[str, Any],
) -> None:
    for idx, metric in metrics.items():
        reason = reasons_responses.get(idx)
        if isinstance(reason, HallucinationScoreReason):
            metric.reason = reason.reason
        else:
            logger.error("Error parsing hallucination reason: %s", reason)
            metric.reason = "error_parsing_reason"
