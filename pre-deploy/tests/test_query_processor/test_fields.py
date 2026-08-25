import unittest
import json
import uuid
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.request_dict import RequestDict, copy_request_dict
from pre_deploy.query_processor.client import QueryProcessorClient
from pre_deploy.query_processor.deepeval import DeepEvalClient
from pre_deploy.query_processor.environment import AWSEnvironment


class TestSQSFields(unittest.TestCase):
    """Unit tests to sanity check fields passed to SQS in query_processor modules."""

    def setUp(self):
        """Set up test fixtures."""
        self.run_id = str(uuid.uuid4())

        # Mock AWS environment and DynamoDB
        self.mock_environment = Mock(spec=AWSEnvironment)
        self.mock_environment.sqs_client = Mock()
        self.mock_environment.sqs_queue_url = 'https://sqs.us-east-1.amazonaws.com/123456789/test-queue'
        self.mock_environment.sqs_client.send_message.return_value = {"MessageId": "msg-123"}

        self.query_dict = {
            '0': 'What is AI?',
            '1': 'Explain machine learning',
            '2': 'What is deep learning?'
        }

        self.request_dict = RequestDict(
            username='test_user',
            metric_name='test_metric',
            metric_phase='test_phase',
            run_id=self.run_id,
            num_eval=len(self.query_dict),
            max_task=3,
            model_name='gpt-4',
            model_provider='openai',
            project_id=None,
            project_version=None,
            dataset_version_id='null',
            system_prompt='You are a helpful assistant.',
            enable_search=True,
            search_collection='test_collection',
            search_top_k=5,
            model_temperature=0.7,
            model_top_p=0.9,
            model_top_k=40,
            model_max_tokens=2048
        )

    @contextmanager
    def _patched_sqs_dependencies(self):
        with patch('pre_deploy.query_processor.client.write_to_query_processor_table'), \
             patch(
                 'pre_deploy.query_processor.client.read_model_config_ddb_table',
                 return_value=[{"id": "gpt-4", "provider": "openai", "is_active": True}]
             ):
            yield

    def _send_and_collect_messages(self, request_dict=None, query_dict=None):
        request_dict = request_dict or self.request_dict
        query_dict = query_dict or self.query_dict

        with self._patched_sqs_dependencies():
            client = QueryProcessorClient(
                request_dict,
                self.mock_environment,
                query_dict
            )
            client.send_queries_to_sqs()

        return [
            json.loads(call[1]['MessageBody'])
            for call in self.mock_environment.sqs_client.send_message.call_args_list
        ]

    def test_sqs_message_contains_representative_core_fields(self):
        """Test representative request/query/model/search/system fields in SQS payload."""
        messages = self._send_and_collect_messages()
        self.assertEqual(len(messages), len(self.query_dict))

        for message_dict in messages:
            self.assertEqual(message_dict['username'], 'test_user')
            self.assertEqual(message_dict['metric_name'], 'test_metric')
            self.assertEqual(message_dict['metric_phase'], 'test_phase')
            self.assertEqual(message_dict['run_id'], self.run_id)
            self.assertEqual(message_dict['model_name'], 'gpt-4')
            self.assertEqual(message_dict['model_provider'], 'openai')
            self.assertTrue(message_dict['enable_search'])
            self.assertEqual(message_dict['search_collection'], 'test_collection')
            self.assertEqual(message_dict['system_prompt'], 'You are a helpful assistant.')
            self.assertIn(message_dict['query_number'], self.query_dict)
            self.assertEqual(message_dict['query'], self.query_dict[message_dict['query_number']])
            self.assertEqual(message_dict['total_items'], len(self.query_dict))
            self.assertEqual(message_dict['num_eval'], len(self.query_dict))
            self.assertEqual(message_dict['max_task'], 3)
            self.assertTrue(uuid.UUID(message_dict['id']))
            self.assertIsNotNone(message_dict['version_id'])

    def test_sqs_message_contains_dataset_version_id_when_provided(self):
        """Test that dataset_version_id is mapped into version_id in SQS payload."""
        dataset_version_id = 'dataset-v1'
        self.request_dict.dataset_version_id = dataset_version_id
        messages = self._send_and_collect_messages()

        for message_dict in messages:
            self.assertEqual(message_dict['version_id'], dataset_version_id)

    def test_sqs_message_optional_fields_default_when_not_provided(self):
        """Test optional request fields default to None while required fields remain present."""
        minimal_request = RequestDict(
            username='test_user',
            metric_name='test_metric',
            metric_phase='test_phase',
            run_id=str(uuid.uuid4()),
            project_id='test-project'
        )

        messages = self._send_and_collect_messages(
            request_dict=minimal_request,
            query_dict={'0': 'Test query'}
        )

        self.assertEqual(len(messages), 1)
        message_dict = messages[0]
        self.assertEqual(message_dict['username'], 'test_user')
        self.assertEqual(message_dict['query'], 'Test query')
        self.assertIsNone(message_dict['model_name'])
        self.assertIsNone(message_dict['project_version'])
        self.assertIsNotNone(message_dict['version_id'])

        call_kwargs = self.mock_environment.sqs_client.send_message.call_args[1]
        self.assertEqual(call_kwargs['QueueUrl'], self.mock_environment.sqs_queue_url)

    def test_deepeval_batch_generate_passes_dataset_version_id(self):
        """Test that DeepEvalClient passes dataset_version_id to QueryProcessorClient."""
        with patch('pre_deploy.query_processor.deepeval.copy_request_dict') as mock_copy, \
             patch('pre_deploy.query_processor.deepeval.QueryProcessorClient') as mock_query_processor_client:

            copied_request = self.request_dict
            mock_copy.return_value = copied_request

            qp_instance = mock_query_processor_client.return_value
            qp_instance.fetch_responses.return_value = {}

            client = DeepEvalClient(
                self.request_dict,
                self.mock_environment,
                timeout_seconds=30
            )

            prompts = {'0': 'Prompt 1', '1': 'Prompt 2'}
            dataset_version_id = 'dataset-v1'
            copied_request.dataset_version_id = dataset_version_id
            client.batch_generate(prompts)

            self.assertEqual(copied_request.dataset_version_id, dataset_version_id)
            mock_query_processor_client.assert_called_once_with(
                copied_request,
                self.mock_environment,
                prompts,
                timeout_seconds=30
            )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
