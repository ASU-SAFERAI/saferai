import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.accuracy_with_reference import accuracy_with_reference_batch_generate
from tests.utils import create_eval_dataset_for_testing

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestAccuracy(unittest.TestCase):
    def test_accuracy_incorrect_response(self):
        run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Who is the third person in the line?",
                "actual_output": "Gabby.",
                "expected_output": "Adam."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})

        # Call the function
        results = accuracy_with_reference_batch_generate(
            evaluator_info=run_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            verbose_mode=False
        )

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        results_json = results.to_dict()

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertAlmostEqual(item['score'], 0.0, places=2, msg="Score should be close to 0.0")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertFalse(item['success'], "Success should be False")

    def test_accuracy_correct_response(self):
        run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Who is the third person in the line? Who is the fourth person in the line?",
                "actual_output": "Gabby is the third person and Adam is the fourth.",
                "expected_output": "Gabby is the third person and Adam is the fourth."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})

        # Call the function
        results = accuracy_with_reference_batch_generate(
            evaluator_info=run_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            verbose_mode=False
        )

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        results_json = results.to_dict()

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertAlmostEqual(item['score'], 1.0, places=2, msg="Score should be close to 1.0")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertTrue(item['success'], "Success should be True")

    def test_accuracy_partially_correct_response(self):
        run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Who is the third person in the line? Who is the fourth person in the line?",
                "actual_output": "Gabby is the third person. I am not sure about the fourth.",
                "expected_output": "Gabby is the third person and Adam is the fourth."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})

        # Call the function
        results = accuracy_with_reference_batch_generate(
            evaluator_info=run_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            verbose_mode=False
        )

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        results_json = results.to_dict()

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertTrue(item['score'] < 0.5, msg="Score should be below 0.5")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertFalse(item['success'], "It should not pass the test")


if __name__ == '__main__':
    unittest.main()
