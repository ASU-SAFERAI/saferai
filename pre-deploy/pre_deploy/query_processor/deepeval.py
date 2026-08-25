from deepeval.models import DeepEvalBaseLLM
from deepeval.models.llms.utils import trim_and_load_json
from pydantic import BaseModel

from typing import Any, Dict, List, Optional, Type, Union
import logging

from . import RequestDict, QueryProcessorClient, copy_request_dict
from .environment import AWSEnvironment

logger = logging.getLogger(__name__)


class DeepEvalClient(DeepEvalBaseLLM):
    def __init__(
        self,
        request_dict: RequestDict,
        environment: AWSEnvironment,
        timeout_seconds: int = 15 * 60
    ):
        """DeepEval-compatible LLM wrapper using the SQS Query Processor.
        Args:
            request_dict: Base request configuration (copied per batch).
            timeout_seconds: Timeout for waiting for responses from the Query Processor.
        """
        self.model_name = request_dict.model_name or request_dict.project_id or "UnknownModel"
        self.model = None
        self.request_dict = request_dict
        self.environment = environment
        self._timeout_seconds = timeout_seconds
        self._batcher = None

    def load_model(self, *args, **kwargs) -> None:
        raise NotImplementedError("DeepEvalClient does not support load_model(). Use batch_generate() instead.")

    def generate(self, prompt: str) -> str:
        raise NotImplementedError("DeepEvalClient does not support generate(). Use batch_generate() instead.")

    async def a_generate(self, prompt: str) -> str:
        raise NotImplementedError("DeepEvalClient does not support a_generate(). Use batch_generate() instead.")

    @staticmethod
    def _normalize_prompts(prompts: Union[Dict[str, str], List[str]]) -> Dict[str, str]:
        if isinstance(prompts, list):
            return {str(i): prompts[i] for i in range(len(prompts))}
        return prompts

    def enqueue_batch(
        self,
        prompts: Union[Dict[str, str], List[str]],
        create_new_run_id: bool = False,
        force_rerun: bool = False,
    ) -> Dict[str, Any]:
        """Send prompts to Query Processor and return immediately."""
        if create_new_run_id:
            self.request_dict = copy_request_dict(self.request_dict)

        prompts = self._normalize_prompts(prompts)
        client = QueryProcessorClient(
            self.request_dict,
            self.environment,
            prompts,
            timeout_seconds=self._timeout_seconds,
        )
        logger.info(
            f"Enqueueing {len(prompts)} queries to Query Processor with run_id: {self.request_dict.run_id}"
        )
        metadata = client.enqueue_phase(force_rerun=force_rerun)
        metadata["query_keys"] = list(prompts.keys())
        return metadata

    def finalize_batch(
        self,
        prompts: Union[Dict[str, str], List[str]],
        schema: Optional[Type[BaseModel]] = None,
        allow_partial: bool = False,
    ) -> Dict[str, Any]:
        """Fetch phase responses without enqueueing new prompts.

        Returns:
            If incomplete and allow_partial=False: status dict with total_items, completed_items, pending_items, is_complete.
            Otherwise: response dict (parsed if schema provided) or metadata.
        """
        prompts = self._normalize_prompts(prompts)
        client = QueryProcessorClient(
            self.request_dict,
            self.environment,
            prompts,
            timeout_seconds=self._timeout_seconds,
        )
        result = client.finalize_phase()

        if not allow_partial and not result["is_complete"]:
            # Return status dict instead of raising, for transparent polling.
            return {
                "total_items": result["total_items"],
                "completed_items": result["completed_items"],
                "pending_items": result["pending_items"],
                "is_complete": result["is_complete"],
                "run_id": self.request_dict.run_id,
                "metric_phase": self.request_dict.metric_phase,
            }

        responses = result["responses"]
        if schema is None:
            return responses

        return {
            idx: self.parse_response(response, schema)
            for idx, response in responses.items()
            if response is not None
        }

    def batch_generate(self, prompts: Union[Dict[str, str], List[str]], schema: Optional[Type[BaseModel]] = None,
                       create_new_run_id: bool = False) -> Dict[str, Any]:
        """Only operation supported for the SQS Batch Query Processor."""
        prompts = self._normalize_prompts(prompts)
        self.enqueue_batch(
            prompts=prompts,
            create_new_run_id=create_new_run_id,
            force_rerun=False,
        )
        client = QueryProcessorClient(
            self.request_dict,
            self.environment,
            prompts,
            timeout_seconds=self._timeout_seconds,
        )
        try:
            client.status_check()
        except TimeoutError as e:
            logger.warning(f"Run incomplete: {e}")
        finally:
            responses = self.finalize_batch(prompts=prompts, schema=schema, allow_partial=True)
            logger.info(f"Fetched {len(responses)} responses out of {len(prompts)} queries.")
        return responses

    def get_model_name(self) -> str:
        return self.model_name

    @staticmethod
    def parse_response(raw_json: str, schema: Optional[Type[BaseModel]]) -> Union[Dict[str, Any], BaseModel]:
        try:
            logger.debug(f"Raw JSON response: {raw_json}")
            json_output = trim_and_load_json(raw_json)
            if schema is not None:
                json_output = schema.model_validate(json_output)
        except Exception as e:
            logger.error(f"Erroneous JSON response: {raw_json}\n"
                         f"Error parsing or validating JSON response: {e}")
            return {}
        else:
            return json_output
