import logging
from typing import Any, Dict, Optional

from ..input import EvalDataset
from ..query_processor import AWSEnvironment, DeepEvalClient, QueryProcessorClient
from ..query_processor.request_dict import RequestDict

logger = logging.getLogger(__name__)


def _prompt_key(idx: str, i: int) -> str:
    return f"{idx}_intention_{i}"

def _extract_idx(key: str) -> str:
    parts = key.rsplit("_intention_", 1)
    if len(parts) == 2:
        return parts[0]
    else:
        raise ValueError(f"Invalid key format: {key}")


def _build_metric_request_dict(
    evaluator_info: RequestDict,
    metric_name: str,
    num_eval: int,
    dataset_version_id: str,
) -> RequestDict:
    request_dict = RequestDict.from_dict(evaluator_info.to_dict())
    request_dict.metric_name = metric_name
    request_dict.metric_phase = ""
    request_dict.num_eval = num_eval
    request_dict.dataset_version_id = dataset_version_id
    return request_dict


def _validate_dataset_version_id(eval_dataset: EvalDataset) -> str:
    dataset_version_id = eval_dataset.metadata.get("dataset_version_id")
    if not isinstance(dataset_version_id, str) or len(dataset_version_id) <= 1:
        raise ValueError(
            "eval_dataset.metadata['dataset_version_id'] must be a string with length > 1 "
            "(use 'null' when no dataset version exists)."
        )
    return dataset_version_id


def _build_metric_model(
    evaluator_info: RequestDict,
    eval_dataset: EvalDataset,
    metric_name: str,
    environment: Optional[AWSEnvironment],
) -> DeepEvalClient:
    dataset_version_id = _validate_dataset_version_id(eval_dataset)

    if environment is None:
        environment = AWSEnvironment(target_account_id=None, role_name=None)

    request_dict = _build_metric_request_dict(
        evaluator_info=evaluator_info,
        metric_name=metric_name,
        num_eval=len(eval_dataset.conversations),
        dataset_version_id=dataset_version_id,
    )
    return DeepEvalClient(request_dict=request_dict, environment=environment)


def get_phase_status(responses: Dict[str, Any], total_items: int) -> Dict[str, Any]:
    """Get phase completion status from response dictionary.

    Args:
        responses: Dict mapping query_number to response (None if not yet received).
        total_items: Total expected items.

    Returns:
        Dict with keys: total_items, completed_items, pending_items, is_complete.
    """
    completed = sum(1 for r in responses.values() if r is not None)
    pending = total_items - completed
    return {
        "completed_items": completed,
        "pending_items": pending,
        "total_items": total_items,
        "is_complete": pending == 0,
    }


def _phase_is_complete(
    model: DeepEvalClient,
    prompts: Dict[str, Any],
    phase: str,
) -> Dict[str, Any]:
    """Check phase completion status for a given prompt set.

    Returns a status dict with total/completed/pending counts and is_complete flag.
    """
    if not prompts:
        return {
            "total_items": 0,
            "completed_items": 0,
            "pending_items": 0,
            "is_complete": True,
        }

    model.request_dict.metric_phase = phase
    status = QueryProcessorClient(
        model.request_dict,
        model.environment,
        prompts,
        timeout_seconds=getattr(model, "_timeout_seconds", 15 * 60),
    ).check_phase_status()

    if not status["is_complete"]:
        logger.info(
            "Phase %s for run_id %s is incomplete: %s/%s",
            phase,
            model.request_dict.run_id,
            status["completed_items"],
            status["completed_items"] + status["pending_items"],
        )
    return status


def _is_incomplete_status(result: Any) -> bool:
    """Return True when a phase result is an incomplete status dict."""
    return isinstance(result, dict) and "is_complete" in result and not result["is_complete"]


def _finalize_phase_or_status(
    model: DeepEvalClient,
    prompts: Dict[str, Any],
    phase: str,
    schema: Any,
) -> Dict[str, Any]:
    """Finalize a phase when complete, else return phase status details."""
    status = _phase_is_complete(model, prompts, phase)
    if not status["is_complete"]:
        return status
    return model.finalize_batch(prompts=prompts, schema=schema, allow_partial=True)
