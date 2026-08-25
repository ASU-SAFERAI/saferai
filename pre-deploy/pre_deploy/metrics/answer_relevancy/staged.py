from typing import Any, Dict, Optional, Union, cast

from deepeval.metrics.answer_relevancy.schema import (
	AnswerRelevancyScoreReason,
	Statements,
	Verdicts,
)

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output.results import MetricsResults
from ...query_processor import AWSEnvironment, DeepEvalClient, RequestDict
from ..utils import _build_metric_model, _finalize_phase_or_status, _is_incomplete_status
from .helpers import (
	METRIC_NAME,
	_apply_reasons,
	_apply_statements,
	_apply_verdicts,
	_build_metrics,
	_build_reasons_prompts_from_verdicts,
	_build_statements_prompts,
	_build_verdicts_prompts_from_statements,
)


def statements_batch_generate(
	model: DeepEvalClient,
	test_cases_dict: Dict[str, Any],
	enqueue_only: bool = True,
	force_rerun: bool = False,
) -> Union[Dict[str, Any], Dict[str, Statements]]:
	"""Run only the statements stage."""
	prompts = _build_statements_prompts(test_cases_dict)
	model.request_dict.metric_phase = "statements"

	if enqueue_only:
		return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
	return _finalize_phase_or_status(model, prompts, "statements", Statements)


def verdicts_batch_generate(
	model: DeepEvalClient,
	test_cases_dict: Dict[str, Any],
	enqueue_only: bool = True,
	force_rerun: bool = False,
	statements_responses: Optional[Dict[str, Statements]] = None,
) -> Union[Dict[str, Any], Dict[str, Verdicts]]:
	"""Run only the verdicts stage. Reads statements from Query Processor."""
	if statements_responses is None:
		statements_result = statements_batch_generate(
			model=model,
			test_cases_dict=test_cases_dict,
			enqueue_only=False,
		)
		if _is_incomplete_status(statements_result):
			return statements_result
		statements_responses = cast(Dict[str, Statements], statements_result)

	prompts = _build_verdicts_prompts_from_statements(test_cases_dict, statements_responses)
	model.request_dict.metric_phase = "verdicts"

	if enqueue_only:
		return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
	return _finalize_phase_or_status(model, prompts, "verdicts", Verdicts)


def reasons_batch_generate(
	model: DeepEvalClient,
	test_cases_dict: Dict[str, Any],
	threshold: float = 0.5,
	strict_mode: bool = False,
	enqueue_only: bool = True,
	force_rerun: bool = False,
	verdicts_responses: Optional[Dict[str, Verdicts]] = None,
	statements_responses: Optional[Dict[str, Statements]] = None,
) -> Union[Dict[str, Any], Dict[str, AnswerRelevancyScoreReason]]:
	"""Run only the reasons stage. Reads statements and verdicts from Query Processor."""
	if verdicts_responses is None:
		verdicts_result = verdicts_batch_generate(
			model=model,
			test_cases_dict=test_cases_dict,
			enqueue_only=False,
			statements_responses=statements_responses,
		)
		if _is_incomplete_status(verdicts_result):
			return verdicts_result
		verdicts_responses = cast(Dict[str, Verdicts], verdicts_result)

	prompts = _build_reasons_prompts_from_verdicts(
		model=model,
		test_cases_dict=test_cases_dict,
		verdicts_responses=verdicts_responses,
		threshold=threshold,
		strict_mode=strict_mode,
	)
	model.request_dict.metric_phase = "reasons"

	if enqueue_only:
		return model.enqueue_batch(prompts=prompts, force_rerun=force_rerun)
	return _finalize_phase_or_status(model, prompts, "reasons", AnswerRelevancyScoreReason)


def start_answer_relevancy(
	evaluator_info: RequestDict,
	eval_dataset: EvalDataset,
	environment: Optional[AWSEnvironment] = None,
	force_rerun: bool = False,
) -> Dict[str, Any]:
	"""Compatibility wrapper for statements enqueue."""
	model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
	test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
	return cast(
		Dict[str, Any],
		statements_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		enqueue_only=True,
		force_rerun=force_rerun,
		),
	)

def advance_answer_relevancy(
	evaluator_info: RequestDict,
	eval_dataset: EvalDataset,
	phase: str,
	threshold: float = 0.5,
	include_reason: bool = True,
	strict_mode: bool = False,
	environment: Optional[AWSEnvironment] = None,
	force_rerun: bool = False,
) -> Dict[str, Any]:
	"""Advance by enqueueing a target stage or returning prerequisite phase progress."""
	if phase not in {"verdicts", "reasons"}:
		raise ValueError("phase must be one of: verdicts, reasons")
	if phase == "reasons" and not include_reason:
		raise ValueError("Cannot advance to reasons when include_reason=False")

	model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
	test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)

	if phase == "verdicts":
		statements_result = statements_batch_generate(
			model=model,
			test_cases_dict=test_cases_dict,
			enqueue_only=False,
		)
		if _is_incomplete_status(statements_result):
			return cast(Dict[str, Any], statements_result)
		statements_responses = cast(Dict[str, Statements], statements_result)
		return cast(
			Dict[str, Any],
			verdicts_batch_generate(
			model=model,
			test_cases_dict=test_cases_dict,
			enqueue_only=True,
			force_rerun=force_rerun,
			statements_responses=statements_responses,
			),
		)

	statements_result = statements_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		enqueue_only=False,
	)
	if _is_incomplete_status(statements_result):
		return cast(Dict[str, Any], statements_result)
	statements_responses = cast(Dict[str, Statements], statements_result)

	verdicts_result = verdicts_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		enqueue_only=False,
		statements_responses=statements_responses,
	)
	if _is_incomplete_status(verdicts_result):
		return cast(Dict[str, Any], verdicts_result)
	verdicts_responses = cast(Dict[str, Verdicts], verdicts_result)

	return cast(
		Dict[str, Any],
		reasons_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		threshold=threshold,
		strict_mode=strict_mode,
		enqueue_only=True,
		force_rerun=force_rerun,
		verdicts_responses=verdicts_responses,
		),
	)


def finalize_answer_relevancy(
	evaluator_info: RequestDict,
	eval_dataset: EvalDataset,
	threshold: float = 0.5,
	include_reason: bool = True,
	strict_mode: bool = False,
	verbose_mode: bool = False,
	environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
	"""Finalize stages sequentially; return status dict if any required phase is incomplete."""
	model = _build_metric_model(evaluator_info, eval_dataset, METRIC_NAME, environment)
	test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)
	metrics = _build_metrics(model, eval_dataset, threshold, include_reason, strict_mode)

	statements_result = statements_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		enqueue_only=False,
	)
	if _is_incomplete_status(statements_result):
		return statements_result
	statements_responses = cast(Dict[str, Statements], statements_result)
	_apply_statements(metrics, statements_responses)

	verdicts_result = verdicts_batch_generate(
		model=model,
		test_cases_dict=test_cases_dict,
		enqueue_only=False,
		statements_responses=statements_responses,
	)
	if _is_incomplete_status(verdicts_result):
		return verdicts_result
	verdicts_responses = cast(Dict[str, Verdicts], verdicts_result)
	_apply_verdicts(metrics, verdicts_responses)

	if include_reason:
		reasons_result = reasons_batch_generate(
			model=model,
			test_cases_dict=test_cases_dict,
			threshold=threshold,
			strict_mode=strict_mode,
			enqueue_only=False,
			verdicts_responses=verdicts_responses,
		)
		if _is_incomplete_status(reasons_result):
			return reasons_result
		reasons_responses = cast(Dict[str, AnswerRelevancyScoreReason], reasons_result)
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
