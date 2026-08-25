import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.metrics.consistency import consistency_batch_generate
from pre_deploy.query_processor import RequestDict
from tests.utils import create_eval_dataset_for_testing

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestConsistency(unittest.TestCase):

    @patch('pre_deploy.query_processor.QueryProcessorClient.fetch_responses')
    def test_consistency(self, mock_fetch_responses):
        """Test consistency."""
        mock_fetch_responses.return_value = {'1': '{\n"score": 10,\n "reason": "Consistent responses."\n}', '2': '{"score": 10, "reason": "Consistent."}'}

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Response 1: Hi! How can I help?\nResponse 2: Hi! How can I help?\nResponse 3: Hi! How can I help?",
                "id": "1"
            },
            {
                "input": "Hi!",
                "actual_output": "Response 1: Hi! How can I help?\nResponse 2: Hi! How can I help?\nResponse 3: Hi! How can I help?",
                "id": "2"
            },
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        results = consistency_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            threshold=0.5
        )

        results_json = results.to_dict()

        self.assertEqual(len(results_json["results"]), 2)
        self.assertIsNotNone(results_json.get('name'), "Name should be populated")
        self.assertIsNotNone(results_json.get('run_id'), "Run ID should be populated")

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")


if __name__ == '__main__':
    unittest.main()
