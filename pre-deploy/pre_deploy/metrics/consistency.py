from collections import defaultdict
import logging
from typing import Optional, Union, Dict, Any

from .custom import (
    advance_custom,
    custom_batch_generate,
    finalize_custom,
    start_custom,
)
from ..input import Conversation, EvalDataset
from ..output import MetricsResults
from ..query_processor import RequestDict, AWSEnvironment

logger = logging.getLogger(__name__)

CONSISTENCY_EVALUATION_STEPS = [
    "1. Compare the responses to the same question.",
    "2. Check if the responses convey the same meaning and information.",
    "3. Look for any contradictions or inconsistencies between responses.",
    "4. Consider if the responses maintain the same level of detail and accuracy.",
    "5. If responses are consistent in meaning and information, award a high score.",
    "6. If there are minor variations in wording but same meaning, still award a high score.",
    "7. If there are contradictions or significant inconsistencies, reduce the score.",
    "8. Assign a final score based on the level of consistency across responses."
]


def consistency_batch_generate(evaluator_info: RequestDict,
                               eval_dataset: EvalDataset,
                               threshold: float = 0.5,
                               verbose_mode: bool = False,
                               environment: Optional[AWSEnvironment] = None) -> Union[MetricsResults, Dict[str, Any]]:
    """
    Compute consistency evaluation by compiling variations and evaluating them.
    Input eval_dataset should have multiple responses per input identified by original_id.
    e.g. The structure of a conversation in eval_dataset should be like:
    {
            'id': 'example_1',
            'messages': [
                {
                    'sequence': 0,
                    'role': "system",
                    'contents': [{'type': "text", 'content': "<input>"}],
                },
                {
                    'sequence': 1,
                    'role': "user",
                    'contents': [{'type': "text", 'content': "Response 1: <...>\nResponse 2: <...>\nResponse 3: <...>..."}],
                },
            ],
            'metadata': {}
    }
    """
    if environment is None:
        environment = AWSEnvironment(target_account_id=None, role_name=None)

    results = custom_batch_generate(name="consistency",
                                    evaluator_info=evaluator_info,
                                    eval_dataset=eval_dataset,
                                    evaluation_steps=CONSISTENCY_EVALUATION_STEPS,
                                    with_reference=False,
                                    strict_mode=False,
                                    threshold=threshold,
                                    verbose_mode=verbose_mode,
                                    environment=environment)

    return results


def start_consistency(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    return start_custom(
        name="consistency",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        with_reference=False,
        evaluation_steps=CONSISTENCY_EVALUATION_STEPS,
        threshold=threshold,
        strict_mode=False,
        environment=environment,
        force_rerun=force_rerun,
    )


def advance_consistency(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    phase: str,
    threshold: float = 0.5,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Union[Dict[str, Any]]:
    return advance_custom(
        name="consistency",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        phase=phase,
        with_reference=False,
        evaluation_steps=CONSISTENCY_EVALUATION_STEPS,
        threshold=threshold,
        strict_mode=False,
        environment=environment,
        force_rerun=force_rerun,
    )


def finalize_consistency(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    threshold: float = 0.5,
    verbose_mode: bool = False,
    environment: Optional[AWSEnvironment] = None,
) -> Union[MetricsResults, Dict[str, Any]]:
    return finalize_custom(
        name="consistency",
        evaluator_info=evaluator_info,
        eval_dataset=eval_dataset,
        with_reference=False,
        evaluation_steps=CONSISTENCY_EVALUATION_STEPS,
        threshold=threshold,
        strict_mode=False,
        verbose_mode=verbose_mode,
        environment=environment,
    )


def compute_consistency_test_set(eval_dataset: EvalDataset, n: int) -> EvalDataset:

    """Generate a new GoldenTestSet where each input is repeated n times with different IDs."""

    conversations = []

    for convo in eval_dataset.conversations:

        for i in range(n):
            new_id = f"{convo.id}_{i}"
            convo_dict = convo.to_dict()
            convo_dict['id'] = new_id
            convo_dict['metadata']['original_id'] = convo.id
            conversations.append(convo_dict)

    new_eval_dataset_dict = {
        "id": f"{eval_dataset.id}_consistency(rep_{n})",
        "data": conversations,
        "metadata": eval_dataset.metadata
    }

    new_eval_dataset = EvalDataset.from_dict(new_eval_dataset_dict)

    return new_eval_dataset
