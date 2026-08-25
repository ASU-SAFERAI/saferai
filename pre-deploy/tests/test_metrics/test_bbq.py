import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, PropertyMock
from datasets import Dataset

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.bbq import BBQ
from pre_deploy import EvalDataset

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestBBQ(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures that run before each test method."""
        self.run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

    def test_load_bbq_dataset_without_model(self):
        bbq = BBQ(bias_type="Age")
        bbq_data = bbq.load_bbq_data()
        assert len(bbq_data) == 3680

    def test_bbq_metric_attributes(self):
        bbq_1 = BBQ(bias_type="Age", evaluator_info=self.run_info)
        assert bbq_1.bias_type == "Age"

        eval_data = EvalDataset.from_dict({"id": "unittest", "metadata": {"bias_type": "Age"}, "conversations": []})
        bbq_2 = BBQ(data=eval_data, evaluator_info=self.run_info)
        assert bbq_2.bias_type == "Age"

    def test_set_eval_dataset(self):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        bbq.set_eval_dataset(responses={1: "ans0", 2: "cannot tell"})
        assert bbq.data is not None
        assert len(bbq.data.conversations) == 2
        assert bbq.data.conversations[0].id == "1"

    @patch.object(BBQ, 'eval_responses', new_callable=PropertyMock)
    def test_bbq_stats(self, mock_eval_responses):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        mock_eval_responses.return_value = {"1": "ans1", "2": "ans2"}
        bbq.set_eval_dataset(responses={"1": "ans0", "2": "cannot tell"})
        assert bbq.score == 1.0
        assert bbq.refusal_rate == 0.0

    @patch.object(BBQ, 'eval_responses', new_callable=PropertyMock)
    def test_bbq_stats_idk(self, mock_eval_responses):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        mock_eval_responses.return_value = {"1": "ans1", "2": "idk"}
        bbq.set_eval_dataset(responses={"1": "ans0", "2": "cannot tell"})
        assert bbq.score == 0.5
        assert bbq.refusal_rate == 0.5

    def test_dataset_version_id_metadata_and_output_payload(self):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)

        assert bbq.dataset_version_id == "0.0.0"

        bbq_dataset = Dataset.from_dict({
            "context": ["A short context."],
            "question": ["Who is correct?"],
            "ans0": ["Option A"],
            "ans1": ["Option B"],
            "ans2": ["Option C"],
            "answer_info": [{
                "ans0": ["option_a", "groupA"],
                "ans1": ["option_b", "groupB"],
                "ans2": ["unknown", "unknown"]
            }],
            "additional_metadata": [{"stereotyped_groups": ["groupA"]}],
            "context_condition": ["ambig"],
            "question_polarity": ["nonneg"],
            "label": [0]
        })

        bbq.set_eval_dataset(responses={"0": "ans0"}, bbq_dataset=bbq_dataset)
        assert bbq.data is not None
        assert bbq.data.metadata["dataset_version_id"] == "0.0.0"

        with patch.object(bbq.evaluator, "batch_generate", return_value={"0": "ans0"}):
            payload = bbq.results

        assert "dataset_version_id" in payload
        assert "run_id" in payload
        assert payload["dataset_version_id"] == "0.0.0"
        assert payload["run_id"] is not None

    @patch('pre_deploy.metrics.bbq.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_eval_phase(self, mock_enqueue_batch):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        bbq_dataset = Dataset.from_dict({
            "context": ["A short context."],
            "question": ["Who is correct?"],
            "ans0": ["Option A"],
            "ans1": ["Option B"],
            "ans2": ["Option C"],
            "answer_info": [{
                "ans0": ["option_a", "groupA"],
                "ans1": ["option_b", "groupB"],
                "ans2": ["unknown", "unknown"]
            }],
            "additional_metadata": [{"stereotyped_groups": ["groupA"]}],
            "context_condition": ["ambig"],
            "question_polarity": ["nonneg"],
            "label": [0]
        })
        bbq.set_eval_dataset(responses={"0": "ans0"}, bbq_dataset=bbq_dataset)

        mock_enqueue_batch.return_value = {
            "run_id": self.run_info.run_id,
            "metric_phase": "eval",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = bbq.start_batch_generate()

        self.assertEqual(result["metric_phase"], "eval")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "eval")

    @patch('pre_deploy.metrics.bbq.DeepEvalClient.enqueue_batch', autospec=True)
    def test_advance_reenqueues_eval_phase(self, mock_enqueue_batch):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        bbq_dataset = Dataset.from_dict({
            "context": ["A short context."],
            "question": ["Who is correct?"],
            "ans0": ["Option A"],
            "ans1": ["Option B"],
            "ans2": ["Option C"],
            "answer_info": [{
                "ans0": ["option_a", "groupA"],
                "ans1": ["option_b", "groupB"],
                "ans2": ["unknown", "unknown"]
            }],
            "additional_metadata": [{"stereotyped_groups": ["groupA"]}],
            "context_condition": ["ambig"],
            "question_polarity": ["nonneg"],
            "label": [0]
        })
        bbq.set_eval_dataset(responses={"0": "ans0"}, bbq_dataset=bbq_dataset)

        mock_enqueue_batch.return_value = {
            "run_id": self.run_info.run_id,
            "metric_phase": "eval",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = bbq.advance_batch_generate(phase="eval")

        self.assertEqual(result["metric_phase"], "eval")
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.bbq.DeepEvalClient.finalize_batch', autospec=True)
    def test_finalize_returns_results_when_eval_complete(self, mock_finalize_batch):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        bbq_dataset = Dataset.from_dict({
            "context": ["A short context."],
            "question": ["Who is correct?"],
            "ans0": ["Option A"],
            "ans1": ["Option B"],
            "ans2": ["Option C"],
            "answer_info": [{
                "ans0": ["option_a", "groupA"],
                "ans1": ["option_b", "groupB"],
                "ans2": ["unknown", "unknown"]
            }],
            "additional_metadata": [{"stereotyped_groups": ["groupA"]}],
            "context_condition": ["ambig"],
            "question_polarity": ["nonneg"],
            "label": [0]
        })
        bbq.set_eval_dataset(responses={"0": "ans0"}, bbq_dataset=bbq_dataset)

        mock_finalize_batch.return_value = {"0": "ans0"}

        result = bbq.finalize_batch_generate()

        self.assertIsInstance(result, dict)
        self.assertIn("accuracy", result)
        self.assertIn("run_id", result)
        self.assertTrue(mock_finalize_batch.called)

    @patch('pre_deploy.metrics.bbq.DeepEvalClient.finalize_batch', autospec=True)
    def test_finalize_returns_incomplete_status_when_eval_incomplete(self, mock_finalize_batch):
        bbq = BBQ(bias_type="Age", evaluator_info=self.run_info)
        bbq_dataset = Dataset.from_dict({
            "context": ["A short context."],
            "question": ["Who is correct?"],
            "ans0": ["Option A"],
            "ans1": ["Option B"],
            "ans2": ["Option C"],
            "answer_info": [{
                "ans0": ["option_a", "groupA"],
                "ans1": ["option_b", "groupB"],
                "ans2": ["unknown", "unknown"]
            }],
            "additional_metadata": [{"stereotyped_groups": ["groupA"]}],
            "context_condition": ["ambig"],
            "question_polarity": ["nonneg"],
            "label": [0]
        })
        bbq.set_eval_dataset(responses={"0": "ans0"}, bbq_dataset=bbq_dataset)

        mock_finalize_batch.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = bbq.finalize_batch_generate()

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["pending_items"], 1)
        self.assertEqual(result["completed_items"], 0)


if __name__ == '__main__':
    unittest.main()
