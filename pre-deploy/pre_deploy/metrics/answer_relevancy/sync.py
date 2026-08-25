from typing import Any, Dict, Optional, Union, cast

from deepeval.metrics.answer_relevancy.schema import (
	AnswerRelevancyScoreReason,
	Statements,
	Verdicts,
)

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, RequestDict
from ..utils import _build_metric_model
from .helpers import (
	METRIC_NAME,
	_apply_reasons,
	_apply_statements,
	_apply_verdicts,
	_build_metrics,
	_build_reasons_prompts_from_metrics,
	_build_statements_prompts,
	_build_verdicts_prompts_from_metrics,
)


def answer_relevancy_batch_generate(
	evaluator_info: RequestDict,
	eval_dataset: EvalDataset,
	threshold: float = 0.5,
	include_reason: bool = True,
	strict_mode: bool = False,
	verbose_mode: bool = False,
	environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
	"""Backward-compatible synchronous path."""
	model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
	test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
	metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode)

	model.request_dict.metric_phase = "statements"
	statements_responses = model.batch_generate(
		prompts=_build_statements_prompts(test_cases_dict),
		schema=Statements,
	)
	_apply_statements(metrics, statements_responses)

	model.request_dict.metric_phase = "verdicts"
	verdicts_responses = model.batch_generate(
		prompts=_build_verdicts_prompts_from_metrics(test_cases_dict, metrics),
		schema=Verdicts,
	)
	_apply_verdicts(metrics, verdicts_responses)

	if include_reason:
		model.request_dict.metric_phase = "reasons"
		reasons_responses = model.batch_generate(
			prompts=_build_reasons_prompts_from_metrics(test_cases_dict, metrics),
			schema=AnswerRelevancyScoreReason,
		)
		_apply_reasons(metrics, reasons_responses)

	results = MetricsResults(
		name=METRIC_NAME,
		dataset_info=eval_dataset.metadata,
		run_id=model.request_dict.run_id,
		metrics=cast(Any, metrics),
	)

	if not verbose_mode:
		return results
	return results.compute_verbose_output(test_case_dict=cast(Any, test_cases_dict))
