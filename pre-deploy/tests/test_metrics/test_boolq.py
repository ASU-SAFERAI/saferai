import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, PropertyMock
from datasets import Dataset

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.boolq import BoolQ

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestBoolQ(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures that run before each test method."""
        self.run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt-4o", model_provider="openai")

    def test_load_boolq_dataset_without_model(self):
        boolq = BoolQ()
        boolq_data = boolq.load_boolq_data()
        assert len(boolq_data) == 9427

    def test_set_eval_dataset(self):
        boolq = BoolQ(evaluator_info=self.run_info)
        boolq.set_eval_dataset(responses={1: "True", 10: "False"})
        assert boolq.data is not None
        assert len(boolq.data.conversations) == 2
        assert boolq.data.conversations[0].id == "1"
        assert boolq.data.conversations[1].id == "10"

    @patch.object(BoolQ, 'eval_responses', new_callable=PropertyMock)
    def test_boolq_stats_correct(self, mock_eval_responses):
        boolq = BoolQ(evaluator_info=self.run_info)
        mock_eval_responses.return_value = {"1": "True", "10": "False"}
        boolq.set_eval_dataset(responses={1: "This is True because ...", 10: "I don't think this is True..."})
        assert boolq.score == 1.0
        assert boolq.refusal_rate == 0.0

    @patch.object(BoolQ, 'eval_responses', new_callable=PropertyMock)
    def test_boolq_stats_idk(self, mock_eval_responses):
        boolq = BoolQ(evaluator_info=self.run_info)
        mock_eval_responses.return_value = {"1": "True", "10": "idk"}
        boolq.set_eval_dataset(responses={1: "This is True because ...", 10: "I don't have enough information"})
        assert boolq.score == 0.5
        assert boolq.refusal_rate == 0.5

    def test_dataset_version_id_metadata_and_output_payload(self):
        boolq = BoolQ(evaluator_info=self.run_info)

        assert boolq.dataset_version_id == "0.0.0"

        boolq_dataset = Dataset.from_dict({
            "passage": ["Water boils at 100C at sea level."],
            "question": ["Does water boil at 100C"],
            "answer": [True]
        })

        boolq.set_eval_dataset(responses={"0": "True"}, boolq_dataset=boolq_dataset)
        assert boolq.data is not None
        assert boolq.data.metadata["dataset_version_id"] == "0.0.0"

        with patch.object(boolq.evaluator, "batch_generate", return_value={"0": "True"}):
            payload = boolq.results

        assert "dataset_version_id" in payload
        assert "run_id" in payload
        assert payload["dataset_version_id"] == "0.0.0"
        assert payload["run_id"] is not None

    @patch('pre_deploy.metrics.boolq.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_eval_phase(self, mock_enqueue_batch):
        boolq = BoolQ(evaluator_info=self.run_info)
        boolq_dataset = Dataset.from_dict({
            "passage": ["Water boils at 100C at sea level."],
            "question": ["Does water boil at 100C"],
            "answer": [True],
        })
        boolq.set_eval_dataset(responses={"0": "True"}, boolq_dataset=boolq_dataset)

        mock_enqueue_batch.return_value = {
            "run_id": self.run_info.run_id,
            "metric_phase": "eval",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = boolq.start_batch_generate()

        self.assertEqual(result["metric_phase"], "eval")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "eval")

    @patch('pre_deploy.metrics.boolq.DeepEvalClient.enqueue_batch', autospec=True)
    def test_advance_reenqueues_eval_phase(self, mock_enqueue_batch):
        boolq = BoolQ(evaluator_info=self.run_info)
        boolq_dataset = Dataset.from_dict({
            "passage": ["Water boils at 100C at sea level."],
            "question": ["Does water boil at 100C"],
            "answer": [True],
        })
        boolq.set_eval_dataset(responses={"0": "True"}, boolq_dataset=boolq_dataset)

        mock_enqueue_batch.return_value = {
            "run_id": self.run_info.run_id,
            "metric_phase": "eval",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = boolq.advance_batch_generate(phase="eval")

        self.assertEqual(result["metric_phase"], "eval")
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.boolq.DeepEvalClient.finalize_batch', autospec=True)
    def test_finalize_returns_results_when_eval_complete(self, mock_finalize_batch):
        boolq = BoolQ(evaluator_info=self.run_info)
        boolq_dataset = Dataset.from_dict({
            "passage": ["Water boils at 100C at sea level."],
            "question": ["Does water boil at 100C"],
            "answer": [True],
        })
        boolq.set_eval_dataset(responses={"0": "True"}, boolq_dataset=boolq_dataset)

        mock_finalize_batch.return_value = {"0": "True"}

        result = boolq.finalize_batch_generate()

        self.assertIsInstance(result, dict)
        self.assertIn("exact_match_score", result)
        self.assertIn("run_id", result)
        self.assertTrue(mock_finalize_batch.called)

    @patch('pre_deploy.metrics.boolq.DeepEvalClient.finalize_batch', autospec=True)
    def test_finalize_returns_incomplete_status_when_eval_incomplete(self, mock_finalize_batch):
        boolq = BoolQ(evaluator_info=self.run_info)
        boolq_dataset = Dataset.from_dict({
            "passage": ["Water boils at 100C at sea level."],
            "question": ["Does water boil at 100C"],
            "answer": [True],
        })
        boolq.set_eval_dataset(responses={"0": "True"}, boolq_dataset=boolq_dataset)

        mock_finalize_batch.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = boolq.finalize_batch_generate()

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["pending_items"], 1)
        self.assertEqual(result["completed_items"], 0)


if __name__ == '__main__':
    unittest.main()
