import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
import uuid

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.deepeval import DeepEvalClient
from pre_deploy.query_processor.request_dict import RequestDict


class TestDeepEvalClientPhases(unittest.TestCase):
    def setUp(self):
        self.request_dict = RequestDict(
            username="test_user",
            run_id=str(uuid.uuid4()),
            metric_name="test_metric",
            metric_phase="test_phase",
            model_name="test_model",
            model_provider="test_provider",
            dataset_version_id="v1",
        )
        self.environment = Mock()
        self.prompts = {"0": "prompt0", "1": "prompt1"}

    @patch("pre_deploy.query_processor.deepeval.QueryProcessorClient")
    def test_enqueue_batch_returns_metadata(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.enqueue_phase.return_value = {
            "run_id": self.request_dict.run_id,
            "metric_phase": "test_phase",
            "total_items": 2,
            "completed_items": 0,
            "pending_items": 2,
            "is_complete": False,
        }

        client = DeepEvalClient(self.request_dict, self.environment)
        result = client.enqueue_batch(self.prompts)

        self.assertEqual(result["run_id"], self.request_dict.run_id)
        self.assertEqual(result["metric_phase"], "test_phase")
        self.assertEqual(result["query_keys"], ["0", "1"])

    @patch("pre_deploy.query_processor.deepeval.QueryProcessorClient")
    def test_finalize_batch_raises_when_incomplete(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.finalize_phase.return_value = {
            "responses": {"0": "ok"},
            "completed_items": 1,
            "pending_items": 1,
            "is_complete": False,
        }

        client = DeepEvalClient(self.request_dict, self.environment)
        with self.assertRaises(ValueError):
            client.finalize_batch(self.prompts, allow_partial=False)

    @patch("pre_deploy.query_processor.deepeval.QueryProcessorClient")
    def test_finalize_batch_partial_allowed(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        mock_client.finalize_phase.return_value = {
            "responses": {"0": "ok"},
            "completed_items": 1,
            "pending_items": 1,
            "is_complete": False,
        }

        client = DeepEvalClient(self.request_dict, self.environment)
        result = client.finalize_batch(self.prompts, allow_partial=True)

        self.assertEqual(result, {"0": "ok"})

    @patch.object(DeepEvalClient, "finalize_batch")
    @patch.object(DeepEvalClient, "enqueue_batch")
    @patch("pre_deploy.query_processor.deepeval.QueryProcessorClient")
    def test_batch_generate_keeps_compatibility_flow(self, mock_client_cls, mock_enqueue, mock_finalize):
        mock_qp = mock_client_cls.return_value
        mock_finalize.return_value = {"0": "response0", "1": "response1"}

        client = DeepEvalClient(self.request_dict, self.environment)
        result = client.batch_generate(self.prompts)

        self.assertEqual(result, {"0": "response0", "1": "response1"})
        self.assertTrue(mock_enqueue.called)
        self.assertTrue(mock_qp.status_check.called)
        self.assertTrue(mock_finalize.called)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
