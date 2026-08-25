import logging
from typing import Any, Dict, Optional, cast

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy.schema import (
	AnswerRelevancyScoreReason,
	Statements,
	Verdicts,
)
from deepeval.metrics.answer_relevancy.template import AnswerRelevancyTemplate

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...query_processor import DeepEvalClient
from ..utils import _phase_is_complete

logger = logging.getLogger(__name__)

METRIC_NAME = "answer_relevancy"


def _validate_dataset_version_id(eval_dataset: EvalDataset) -> str:
	dataset_version_id = eval_dataset.metadata.get("dataset_version_id")
	if not isinstance(dataset_version_id, str) or len(dataset_version_id) <= 1:
		raise ValueError(
			"eval_dataset.metadata['dataset_version_id'] must be a string with length > 1 "
			"(use 'null' when no dataset version exists)."
		)
	return dataset_version_id


def _build_metrics(
	model: DeepEvalClient,
	eval_dataset: EvalDataset,
	threshold: float,
	include_reason: bool,
	strict_mode: bool,
) -> Dict[str, AnswerRelevancyMetric]:
	return {
		convo.id: AnswerRelevancyMetric(
			model=model,
			threshold=threshold,
			include_reason=include_reason,
			async_mode=False,
			strict_mode=strict_mode,
			verbose_mode=False,
		)
		for convo in eval_dataset.conversations
	}


def _build_statements_prompts(test_cases_dict: Dict[str, Any]) -> Dict[str, str]:
	return {
		idx: AnswerRelevancyTemplate.generate_statements(actual_output=test_case.actual_output)
		for idx, test_case in test_cases_dict.items()
	}


def _build_verdicts_prompts_from_statements(
	test_cases_dict: Dict[str, Any],
	statements_responses: Dict[str, Statements],
) -> Dict[str, str]:
	prompts: Dict[str, str] = {}
	for idx, test_case in test_cases_dict.items():
		statements = statements_responses.get(idx)
		if isinstance(statements, Statements):
			prompts[idx] = AnswerRelevancyTemplate.generate_verdicts(
				input=test_case.input,
				statements=cast(Any, statements.statements),
			)
	return prompts


def _build_verdicts_prompts_from_metrics(
	test_cases_dict: Dict[str, Any],
	metrics: Dict[str, AnswerRelevancyMetric],
) -> Dict[str, str]:
	return {
		idx: AnswerRelevancyTemplate.generate_verdicts(
			input=test_cases_dict[idx].input,
			statements=cast(Any, metric.statements),
		)
		for idx, metric in metrics.items()
		if metric.statements
	}


def _score_from_verdicts(
	model: DeepEvalClient,
	verdicts_response: Verdicts,
	threshold: float,
	strict_mode: bool,
) -> float:
	metric = AnswerRelevancyMetric(
		model=model,
		threshold=threshold,
		include_reason=True,
		async_mode=False,
		strict_mode=strict_mode,
		verbose_mode=False,
	)
	metric.verdicts = verdicts_response.verdicts
	return metric._calculate_score()


def _build_reasons_prompts_from_verdicts(
	model: DeepEvalClient,
	test_cases_dict: Dict[str, Any],
	verdicts_responses: Dict[str, Verdicts],
	threshold: float,
	strict_mode: bool,
) -> Dict[str, str]:
	prompts: Dict[str, str] = {}
	for idx, test_case in test_cases_dict.items():
		verdicts = verdicts_responses.get(idx)
		if not isinstance(verdicts, Verdicts):
			continue

		irrelevant_statements = []
		for verdict in verdicts.verdicts:
			if verdict.verdict.strip().lower() == "no":
				irrelevant_statements.append(verdict.reason)

		score = _score_from_verdicts(model, verdicts, threshold=threshold, strict_mode=strict_mode)
		prompts[idx] = AnswerRelevancyTemplate.generate_reason(
			irrelevant_statements=irrelevant_statements,
			input=test_case.input,
			score=round(score, 2),
		)
	return prompts


def _build_reasons_prompts_from_metrics(
	test_cases_dict: Dict[str, Any],
	metrics: Dict[str, AnswerRelevancyMetric],
) -> Dict[str, str]:
	prompts: Dict[str, str] = {}
	for idx, metric in metrics.items():
		irrelevant_statements = []
		for verdict in metric.verdicts:
			if verdict.verdict.strip().lower() == "no":
				irrelevant_statements.append(verdict.reason)

		prompts[idx] = AnswerRelevancyTemplate.generate_reason(
			irrelevant_statements=irrelevant_statements,
			input=test_cases_dict[idx].input,
			score=round(cast(float, metric.score), 2),
		)
	return prompts


def _apply_statements(
	metrics: Dict[str, AnswerRelevancyMetric],
	statements_responses: Dict[str, Statements],
) -> None:
	for idx, metric in metrics.items():
		if idx in statements_responses and isinstance(statements_responses[idx], Statements):
			metric.statements = statements_responses[idx].statements


def _apply_verdicts(
	metrics: Dict[str, AnswerRelevancyMetric],
	verdicts_responses: Dict[str, Verdicts],
) -> None:
	for idx, metric in metrics.items():
		if idx in verdicts_responses and isinstance(verdicts_responses[idx], Verdicts):
			metric.verdicts = verdicts_responses[idx].verdicts
			metric.score = metric._calculate_score()
			metric.success = (metric.score >= metric.threshold) if metric.score else False


def _apply_reasons(
	metrics: Dict[str, AnswerRelevancyMetric],
	reasons_responses: Dict[str, AnswerRelevancyScoreReason],
) -> None:
	for idx, metric in metrics.items():
		if idx in reasons_responses and isinstance(reasons_responses[idx], AnswerRelevancyScoreReason):
			metric.reason = reasons_responses[idx].reason
		else:
			logger.error("Error parsing answer relevancy reason for idx=%s", idx)
			metric.reason = "error_parsing_reason"
