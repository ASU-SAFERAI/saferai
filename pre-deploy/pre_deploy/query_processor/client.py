from typing import Dict, Any, Optional
import time
import json
import logging
import uuid

from .ddb_handler import (
    write_to_query_processor_table,
    read_responses_from_query_processor_table,
    read_model_config_ddb_table
)
from .request_dict import RequestDict
from .environment import AWSEnvironment
from .alert_manager import AlertManager
from .logs import LogEvent
from .ddb_utils import Encoder

logger = logging.getLogger(__name__)
log_event = LogEvent()


class QueryProcessorClient:
    def __init__(self, request_dict: RequestDict, environment: AWSEnvironment,
                 query_dict: Dict, timeout_seconds: int = 15 * 60):
        """
        Initialize the QueryClient with request configuration and queries.

        Args:
            request_dict: Configuration for the request
            query_dict: A dictionary containing query ids and queries
        """
        self.run_id = request_dict.run_id
        self.request_dict = request_dict
        self.environment = environment
        self.query_dict = query_dict
        self.total_items = len(query_dict)
        self.timeout_seconds = timeout_seconds
        self.response_dict = None
        self.alert_manager = AlertManager(aws_environment=environment)

    def send_queries_to_sqs(self) -> str:
        self._check_request_dict_fields()
        self._check_model_name_and_provider(model_name=self.request_dict.model_name,
                                            model_provider=self.request_dict.model_provider,
                                            project_id=self.request_dict.project_id)

        total_items = len(self.query_dict)
        logger.debug(log_event.format("Processing_Queries", total_items=total_items))

        for index, query in self.query_dict.items():
            individual_query_dict = self.request_dict.to_dict()
            individual_query_dict['query'] = query
            individual_query_dict["query_number"] = index
            individual_query_dict['id'] = str(uuid.uuid4())
            individual_query_dict['total_items'] = total_items  # Include total number of items
            individual_query_dict.pop('dataset_version_id', None)  # Remove dataset_version_id if it exists, will be added as version_id
            individual_query_dict['version_id'] = self.request_dict.dataset_version_id

            try:
                # Write to DynamoDB
                write_to_query_processor_table(self.environment, individual_query_dict, self.alert_manager)
                # Send message to SQS
                self._send_message_to_sqs(individual_query_dict)
            except Exception as e:
                logger.error(log_event.format("Error_in_Query_Processing", query_number=index, error=str(e)))
                raise

    def fetch_responses(self) -> Dict[str, Any]:
        if self.response_dict is not None:
            return self.response_dict
        else:
            return read_responses_from_query_processor_table(
                self.environment,
                self.run_id,
                self.request_dict.metric_phase,
                self.alert_manager
            )

    def status_check(self) -> bool:
        """
        Check the status of the run_id and return the number of completed and pending items.
        Default timeout is set to 300 seconds (5 minutes).
        """
        logger.debug(log_event.format("Run_Status_Check", run_id=self.run_id,
                                      metric_name=self.request_dict.metric_name,
                                      metric_phase=self.request_dict.metric_phase))

        start_time = time.time()

        while True:
            run_status = self.run_id_status()

            if run_status['pending_items'] == 0:
                logger.info(log_event.format("Run_Completed", run_id=self.run_id,
                                             metric_name=self.request_dict.metric_name,
                                             metric_phase=self.request_dict.metric_phase))
                break

            current_time = time.time()

            if current_time - start_time > self.timeout_seconds:
                logger.warning(log_event.format("Run_Timeout", run_id=self.run_id,
                                                 metric_name=self.request_dict.metric_name,
                                                 metric_phase=self.request_dict.metric_phase,
                                                 timeout_seconds=self.timeout_seconds))
                raise TimeoutError(f"Run {self.run_id} did not complete within {self.timeout_seconds} seconds.")

            time.sleep(5)
            logger.debug(log_event.format("Run_Status_Check_Progress", run_id=self.run_id,
                                           metric_name=self.request_dict.metric_name,
                                           metric_phase=self.request_dict.metric_phase,
                                           completed_items=run_status['completed_items'],
                                           pending_items=run_status['pending_items']))

    def run_id_status(self) -> Dict[str, int]:
        """
        Check the status of the run_id and return the number of completed and pending items.

        Returns:
            A dictionary with total items, completed items, and pending items.
        """
        logger.debug(log_event.format("Fetching_status_for_run", run_id=self.run_id,
                                      metric_name=self.request_dict.metric_name,
                                      metric_phase=self.request_dict.metric_phase))

        response_dict = self.fetch_responses()
        completed_items = sum([response is not None for _, response in response_dict.items()])
        pending_items = self.total_items - completed_items

        if pending_items == 0:
            self.response_dict = response_dict

        return {
            "total_items": self.total_items,
            "completed_items": completed_items,
            "pending_items": pending_items
        }

    def enqueue_phase(self, force_rerun: bool = False) -> Dict[str, Any]:
        """Enqueue queries for a phase without waiting for responses.

        Args:
            force_rerun: If True, enqueue even if phase is already complete. Defaults to False.

        Returns:
            Dict with run_id, metric_phase, total_items, completed_items, pending_items, is_complete.
        """
        # Check for existing phase responses (idempotency)
        try:
            status = self.check_phase_status()
            if status["is_complete"] and not force_rerun:
                logger.info(log_event.format("Phase_Already_Complete", run_id=self.run_id,
                                            metric_name=self.request_dict.metric_name,
                                            metric_phase=self.request_dict.metric_phase))
                return {
                    "run_id": self.run_id,
                    "metric_phase": self.request_dict.metric_phase,
                    "total_items": self.total_items,
                    "completed_items": status["completed_items"],
                    "pending_items": status["pending_items"],
                    "is_complete": status["is_complete"],
                }
        except Exception:
            # No existing responses; proceed with enqueue
            pass

        # Send queries without waiting
        self.send_queries_to_sqs()
        logger.info(log_event.format("Phase_Enqueued", run_id=self.run_id,
                                    metric_name=self.request_dict.metric_name,
                                    metric_phase=self.request_dict.metric_phase,
                                    total_items=self.total_items))

        # Get current status after enqueueing for transparency
        enqueue_status = self.check_phase_status()
        return {
            "run_id": self.run_id,
            "metric_phase": self.request_dict.metric_phase,
            "total_items": self.total_items,
            "completed_items": enqueue_status["completed_items"],
            "pending_items": enqueue_status["pending_items"],
            "is_complete": enqueue_status["is_complete"]
        }

    def check_phase_status(self) -> Dict[str, Any]:
        """Check phase completion status without enqueueing anything.

        Returns:
            Dict with total_items, completed_items, pending_items, is_complete.
        """
        # Import lazily to avoid query_processor <-> metrics import cycle at module load time.
        from ..metrics.utils import get_phase_status

        try:
            response_dict = self.fetch_responses()
            status = get_phase_status(response_dict, self.total_items)
            logger.debug(log_event.format("Phase_Status_Check", run_id=self.run_id,
                                         metric_phase=self.request_dict.metric_phase,
                                         **status))
            return status
        except Exception as e:
            logger.debug(log_event.format("Phase_Status_Check_No_Items", run_id=self.run_id,
                                         metric_phase=self.request_dict.metric_phase))
            return {
                "total_items": self.total_items,
                "completed_items": 0,
                "pending_items": self.total_items,
                "is_complete": False,
            }

    def finalize_phase(self) -> Dict[str, Any]:
        """Read completed responses for a phase without enqueueing.

        Returns:
            Dict with responses, total_items, completed_items, pending_items, is_complete.
        """
        # Import lazily to avoid query_processor <-> metrics import cycle at module load time.
        from ..metrics.utils import get_phase_status

        response_dict = self.fetch_responses()
        status = get_phase_status(response_dict, self.total_items)
        self.response_dict = response_dict
        logger.info(log_event.format("Phase_Finalized", run_id=self.run_id,
                                    metric_phase=self.request_dict.metric_phase,
                                    **status))
        return {
            "responses": response_dict,
            **status,
        }

    def _check_request_dict_fields(self):
        if not self.request_dict.username:
            raise ValueError("RequestDict must include a valid username.")
        if not (self.request_dict.model_name and self.request_dict.model_provider) and not self.request_dict.project_id:
            raise ValueError("RequestDict must include either model_name and model_provider, or project_id.")
        if not self.request_dict.run_id:
            raise ValueError("RequestDict must include a valid run_id.")
        if not self.request_dict.metric_name:
            raise ValueError("RequestDict must include a valid metric_name.")
        if not self.request_dict.metric_phase:
            raise ValueError("RequestDict must include a valid metric_phase.")

    def _check_model_name_and_provider(self, model_name: str = None, model_provider: str = None, project_id: str = None):

        if project_id is not None:
            logger.debug(log_event.format(f"Querying CreateAI project {project_id}"))
            return

        models = read_model_config_ddb_table(self.environment, self.alert_manager)

        for model in models:
            if model["id"] == model_name and model["provider"] == model_provider:
                if model['is_active']:
                    logger.debug(log_event.format(f"Model {model_name} with provider {model_provider} is supported"))
                    return
                else:
                    raise Exception(f" Model {model_name} with provider {model_provider} is inactive.")

        raise Exception(f"Model {model_name} with provider {model_provider} is not supported")

    def _send_message_to_sqs(self, message):
        try:
            response = self.environment.sqs_client.send_message(
                QueueUrl=self.environment.sqs_queue_url,
                MessageBody=json.dumps(message, cls=Encoder)
            )
            logger.debug(log_event.format("Message_Sent_to_SQS", message_id=response['MessageId']))
        except Exception as e:
            logger.error(log_event.format("Failed_to_Send_Message_to_SQS", error=str(e)))
            self.alert_manager.notify_error(
                context="query_processor_client_send_message_to_sqs",
                exception=e,
                context_data={"run_id": self.run_id, "metric_phase": self.request_dict.metric_phase},
                log_level="ERROR"
            )
            raise
