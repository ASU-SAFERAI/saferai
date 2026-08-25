import logging
from typing import Any, Dict, List, Optional, Union

from ..query_processor.client import QueryProcessorClient
from ..query_processor.request_dict import RequestDict
from ..query_processor.environment import AWSEnvironment

logger = logging.getLogger(__name__)


def _perturbation_query(input_text: str, perturbation_types: List[str]) -> str:
    perturbation_type_str = " & ".join(perturbation_types)
    return f"""
        Return only a the perturbed output as a string. Do not include explanations, code blocks, double quotes, or extra text.
        For example: Apple.
        Apply {perturbation_type_str} perturbation(s) the following text without changing the original meaning or format.
        Generate 1 perturbed output as a string, don't generate more than 1 item, keep the structure and formatting the same as the original input:
        {input_text}
        """


def _build_perturbation_query_dict(
    query_dict: Dict[str, str],
    perturbation_types: List[str],
) -> Dict[str, str]:
    return {
        key: _perturbation_query(value, perturbation_types)
        for key, value in query_dict.items()
    }


def _resolve_environment(environment: Optional[AWSEnvironment]) -> AWSEnvironment:
    if environment is None:
        return AWSEnvironment(target_account_id=None, role_name=None)
    return environment


def _apply_filter_no_change(
    responses: Dict[str, str],
    original_query_dict: Dict[str, str],
    filter_no_change: bool,
) -> Dict[str, str]:
    if not filter_no_change:
        return responses

    filtered = dict(responses)
    identical_query_ids = set()

    for idx, response in filtered.items():
        if response == original_query_dict[idx]:
            identical_query_ids.add(idx)

    for idx in identical_query_ids:
        del filtered[idx]
        logger.warning(f"Response for query id {idx} is identical to the input. Removed from queries.")

    if filtered == {}:
        logger.warning("No perturbed data generated. Returning empty dictionary.")

    return filtered


def _build_client(
    request_dict: RequestDict,
    query_dict: Dict[str, str],
    perturbation_types: List[str],
    environment: Optional[AWSEnvironment],
) -> QueryProcessorClient:
    perturbation_query_dict = _build_perturbation_query_dict(query_dict, perturbation_types)
    return QueryProcessorClient(
        request_dict=request_dict,
        query_dict=perturbation_query_dict,
        environment=_resolve_environment(environment),
    )


def start_generate_perturbed_data_llm(
    request_dict: RequestDict,
    query_dict: Dict[str, str],
    perturbation_types: List[str],
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    client = _build_client(
        request_dict=request_dict,
        query_dict=query_dict,
        perturbation_types=perturbation_types,
        environment=environment,
    )
    return client.enqueue_phase(force_rerun=force_rerun)


def advance_generate_perturbed_data_llm(
    request_dict: RequestDict,
    query_dict: Dict[str, str],
    perturbation_types: List[str],
    phase: str,
    environment: Optional[AWSEnvironment] = None,
    force_rerun: bool = False,
) -> Dict[str, Any]:
    expected_phase = request_dict.metric_phase
    if phase != expected_phase:
        raise ValueError(f"phase must be: {expected_phase}")

    # Single-phase staged flow: advancing means (re)enqueueing the same phase.
    return start_generate_perturbed_data_llm(
        request_dict=request_dict,
        query_dict=query_dict,
        perturbation_types=perturbation_types,
        environment=environment,
        force_rerun=force_rerun,
    )


def finalize_generate_perturbed_data_llm(
    request_dict: RequestDict,
    query_dict: Dict[str, str],
    perturbation_types: List[str],
    filter_no_change: bool = True,
    environment: Optional[AWSEnvironment] = None,
) -> Dict[str, str]:
    client = _build_client(
        request_dict=request_dict,
        query_dict=query_dict,
        perturbation_types=perturbation_types,
        environment=environment,
    )

    phase_result = client.finalize_phase()
    logger.info(f"Fetched {phase_result['completed_items']} responses out of {len(query_dict)} queries.")

    if not phase_result["is_complete"]:
        return phase_result

    responses = phase_result["responses"]
    return _apply_filter_no_change(
        responses=responses,
        original_query_dict=query_dict,
        filter_no_change=filter_no_change,
    )


def generate_perturbed_data_llm(request_dict: RequestDict,
                                query_dict: Dict[str, str],
                                perturbation_types: List[str],
                                filter_no_change: bool = True,
                                environment: Optional[AWSEnvironment] = None) -> Dict[str, str]:
    client = _build_client(
        request_dict=request_dict,
        query_dict=query_dict,
        perturbation_types=perturbation_types,
        environment=environment,
    )

    client.send_queries_to_sqs()

    try:
        client.status_check()
    except TimeoutError as e:
        logger.warning(f"Run incomplete: {e}")
    finally:
        responses = client.fetch_responses()
        logger.info(f"Fetched {len(responses)} responses out of {len(query_dict)} queries.")

    return _apply_filter_no_change(
        responses=responses,
        original_query_dict=query_dict,
        filter_no_change=filter_no_change,
    )
