import logging
from typing import Optional, Union, Dict, Any

from ..evaluation_steps.accuracy_with_reference import (
    ACCURACY_WITH_REFERENCE_EVALUATION_STEPS,
    ACCURACY_WITH_REFERENCE_RUBRICS
)
from .custom import (
    advance_custom,
    custom_batch_generate,
    finalize_custom,
    start_custom,
)
from ..input import EvalDataset
from ..query_processor import RequestDict, AWSEnvironment
from ..output import MetricsResults

logger = logging.getLogger(__name__)


def accuracy_with_reference_batch_generate(evaluator_info: RequestDict,
                                           eval_dataset: EvalDataset,
                                           threshold: float = 0.5,
                                           verbose_mode: bool = False,
                                           environment: Optional[AWSEnvironment] = None) -> Union[MetricsResults, Dict[str, Any]]:
    if environment is None:
        environment = AWSEnvironment(target_account_id=None, role_name=None)

    results = custom_batch_generate(name="accuracy",
                                    evaluator_info=evaluator_info,
                                    eval_dataset=eval_dataset,
                                    evaluation_steps=ACCURACY_WITH_REFERENCE_EVALUATION_STEPS,
                                    rubrics=ACCURACY_WITH_REFERENCE_RUBRICS,
                                    with_reference=True,
                                    strict_mode=False,
                                    threshold=threshold,
                                    verbose_mode=verbose_mode,
                                    environment=environment)

    return results


def start_accuracy_with_reference(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    return start_custom(
        name="accuracy",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        with_reference=True,
        evaluation_steps=ACCURACY_WITH_REFERENCE_EVALUATION_STEPS,
        rubrics=ACCURACY_WITH_REFERENCE_RUBRICS,
        threshold=threshold,
        strict_mode=False,
        environment=environment,
        force_rerun=force_rerun,
    )


def advance_accuracy_with_reference(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Union[Dict[str, Any], bool]:
    return advance_custom(
        name="accuracy",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        phase=phase,
        with_reference=True,
        evaluation_steps=ACCURACY_WITH_REFERENCE_EVALUATION_STEPS,
        rubrics=ACCURACY_WITH_REFERENCE_RUBRICS,
        threshold=threshold,
        strict_mode=False,
        environment=environment,
        force_rerun=force_rerun,
    )


def finalize_accuracy_with_reference(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    return finalize_custom(
        name="accuracy",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        with_reference=True,
        evaluation_steps=ACCURACY_WITH_REFERENCE_EVALUATION_STEPS,
        rubrics=ACCURACY_WITH_REFERENCE_RUBRICS,
        threshold=threshold,
        strict_mode=False,
        verbose_mode=verbose_mode,
        environment=environment,
    )
