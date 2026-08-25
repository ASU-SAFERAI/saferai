import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import uuid

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.client import QueryProcessorClient
from pre_deploy.query_processor.request_dict import RequestDict
from pre_deploy.metrics.utils import get_phase_status


class TestPhaseStatus(unittest.TestCase):
    """Test phase status utilities."""

    def test_get_phase_status_all_complete(self):
        """Test status when all items are complete."""
        responses = {"0": "response_0", "1": "response_1", "2": "response_2"}
        status = get_phase_status(responses, total_items=3)
        self.assertEqual(status["completed_items"], 3)
        self.assertEqual(status["pending_items"], 0)
        self.assertTrue(status["is_complete"])

    def test_get_phase_status_all_pending(self):
        """Test status when no items are complete."""
        responses = {"0": None, "1": None, "2": None}
        status = get_phase_status(responses, total_items=3)
        self.assertEqual(status["completed_items"], 0)
        self.assertEqual(status["pending_items"], 3)
        self.assertFalse(status["is_complete"])

    def test_get_phase_status_partial(self):
        """Test status with some items complete."""
        responses = {"0": "response_0", "1": None, "2": "response_2"}
        status = get_phase_status(responses, total_items=3)
        self.assertEqual(status["completed_items"], 2)
        self.assertEqual(status["pending_items"], 1)
        self.assertFalse(status["is_complete"])

    def test_get_phase_status_empty(self):
        """Test status with no responses."""
        responses = {}
        status = get_phase_status(responses, total_items=3)
        self.assertEqual(status["completed_items"], 0)
        self.assertEqual(status["pending_items"], 3)
        self.assertFalse(status["is_complete"])


class TestPhaseOperations(unittest.TestCase):
    """Test enqueue_phase and finalize_phase operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.request_dict = RequestDict(
            username="test_user",
            run_id=str(uuid.uuid4()),
            metric_name="test_metric",
            metric_phase="test_phase",
            model_name="test_model",
            model_provider="test_provider",
            dataset_version_id="v1"
        )
        self.environment = Mock()
        self.query_dict = {"0": "query0", "1": "query1", "2": "query2"}

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.send_queries_to_sqs')
    @patch('pre_deploy.query_processor.client.QueryProcessorClient.check_phase_status')
    def test_enqueue_phase_already_complete(self, mock_check_status, mock_send):
        """Test enqueue_phase when phase is already complete (idempotency)."""
        mock_check_status.return_value = {
            "completed_items": 3,
            "pending_items": 0,
            "is_complete": True,
        }
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        result = client.enqueue_phase(force_rerun=False)

        # Should return without sending
        self.assertFalse(mock_send.called)
        # Verify status information is included for transparency
        self.assertEqual(result["completed_items"], 3)
        self.assertEqual(result["pending_items"], 0)
        self.assertTrue(result["is_complete"])

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.send_queries_to_sqs')
    @patch('pre_deploy.query_processor.client.QueryProcessorClient.check_phase_status')
    def test_enqueue_phase_force_rerun(self, mock_check_status, mock_send):
        """Test enqueue_phase with force_rerun=True."""
        # On first call, return complete status; on second call (after send), return same status
        mock_check_status.side_effect = [
            {"completed_items": 3, "pending_items": 0, "is_complete": True},  # first call
            {"completed_items": 3, "pending_items": 0, "is_complete": True},  # second call after send
        ]
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        result = client.enqueue_phase(force_rerun=True)

        # Should send despite phase being complete
        self.assertTrue(mock_send.called)
        # Verify status information is included for transparency
        self.assertEqual(result["completed_items"], 3)
        self.assertEqual(result["pending_items"], 0)
        self.assertTrue(result["is_complete"])

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.send_queries_to_sqs')
    @patch('pre_deploy.query_processor.client.QueryProcessorClient.check_phase_status')
    def test_enqueue_phase_not_complete(self, mock_check_status, mock_send):
        """Test enqueue_phase when phase is not yet complete."""
        # First call raises exception (no responses yet), triggering send
        # Second call (after send) returns the status
        mock_check_status.side_effect = [
            Exception("No items found"),  # first call - no items yet
            {"completed_items": 0, "pending_items": 3, "is_complete": False},  # second call - status after send
        ]
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        result = client.enqueue_phase(force_rerun=False)

        # Should send queries
        self.assertTrue(mock_send.called)
        self.assertEqual(result["run_id"], self.request_dict.run_id)
        self.assertEqual(result["metric_phase"], "test_phase")
        self.assertEqual(result["total_items"], 3)
        # Verify status information is included for transparency
        self.assertEqual(result["completed_items"], 0)
        self.assertEqual(result["pending_items"], 3)
        self.assertFalse(result["is_complete"])

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.fetch_responses')
    def test_check_phase_status_success(self, mock_fetch):
        """Test check_phase_status with successful fetch."""
        mock_fetch.return_value = {"0": "response_0", "1": None, "2": "response_2"}
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        status = client.check_phase_status()

        self.assertEqual(status["completed_items"], 2)
        self.assertEqual(status["pending_items"], 1)
        self.assertFalse(status["is_complete"])

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.fetch_responses')
    def test_check_phase_status_no_items(self, mock_fetch):
        """Test check_phase_status when no items exist yet."""
        mock_fetch.side_effect = Exception("No items found")
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        status = client.check_phase_status()

        self.assertEqual(status["completed_items"], 0)
        self.assertEqual(status["pending_items"], 3)
        self.assertFalse(status["is_complete"])

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.fetch_responses')
    def test_finalize_phase(self, mock_fetch):
        """Test finalize_phase reads responses without enqueueing."""
        mock_fetch.return_value = {"0": "response_0", "1": "response_1", "2": "response_2"}
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        result = client.finalize_phase()

        self.assertEqual(result["completed_items"], 3)
        self.assertEqual(result["pending_items"], 0)
        self.assertTrue(result["is_complete"])
        self.assertIn("responses", result)
        self.assertEqual(result["responses"], {"0": "response_0", "1": "response_1", "2": "response_2"})

    @patch('pre_deploy.query_processor.client.QueryProcessorClient.fetch_responses')
    def test_finalize_phase_partial(self, mock_fetch):
        """Test finalize_phase with partial responses."""
        mock_fetch.return_value = {"0": "response_0", "1": None, "2": "response_2"}
        client = QueryProcessorClient(self.request_dict, self.environment, self.query_dict)

        result = client.finalize_phase()

        self.assertEqual(result["completed_items"], 2)
        self.assertEqual(result["pending_items"], 1)
        self.assertFalse(result["is_complete"])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
