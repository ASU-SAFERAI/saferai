import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.metrics.care import CARE
from pre_deploy.query_processor import RequestDict

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestCARE(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures that run before each test method."""
        self.run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

    def test_load_malformed_care_datasets(self):
        """Test loading malformed CARE datasets."""
        # Test with malformed resource mapping
        with self.assertRaises(TypeError):
            care = CARE(evaluator_info=self.run_info, resource_mapping={"category1": "not_a_list"})

        # Test with malformed safety questions
        with self.assertRaises(KeyError):
            care = CARE(evaluator_info=self.run_info, safety_questions={"1": {"question": "test"}})  # missing "category" key

    def test_load_default_care_datasets(self):
        """Test default loading of CARE datasets."""
        care = CARE(evaluator_info=self.run_info)
        self.assertIsNotNone(care.resource_mapping)
        self.assertIsNotNone(care.safety_questions)

    def test_initialization_with_default_values(self):
        """Test CARE initialization with default parameters."""
        with patch.object(CARE, 'load_and_check_resource_mapping', return_value={}):
            with patch.object(CARE, 'load_and_check_safety_questions', return_value={}):
                care = CARE(evaluator_info=self.run_info)
                self.assertEqual(care.name, "care")
                self.assertEqual(care.threshold, 0.5)
                self.assertIsNone(care.data)

    def test_initialization_with_custom_values(self):
        """Test CARE initialization with custom parameters."""
        custom_mapping = {"category1": ["resource1", "resource2"]}
        custom_questions = {"1": {"question": "test?", "category": "category1"}}

        care = CARE(
            evaluator_info=self.run_info,
            resource_mapping=custom_mapping,
            safety_questions=custom_questions,
            threshold=0.8
        )

        self.assertEqual(care.resource_mapping, custom_mapping)
        self.assertEqual(care.safety_questions, custom_questions)
        self.assertEqual(care.threshold, 0.8)

    def test_compute_evaluation_steps(self):
        """Test _compute_evaluation_steps static method."""
        resource_mapping = ["resource1", "resource2", "resource3"]

        steps = CARE._compute_evaluation_steps(resource_mapping)

        self.assertEqual(len(steps), 4)  # 1 base step + 3 resource steps
        self.assertIn("risk in the question", steps[0])
        self.assertIn("resource1", steps[1])
        self.assertIn("resource2", steps[2])
        self.assertIn("resource3", steps[3])

    def test_set_eval_dataset(self):
        """Test set_eval_dataset method."""
        care = CARE(evaluator_info=self.run_info,
                    resource_mapping={"test category": ["resource1"]},
                    safety_questions={"1": {"question": "test question", "category": "test category"}})
        responses = {"1": "test response"}
        care.set_eval_dataset(responses=responses)

        if care.data is None:
            self.fail("Expected eval dataset to be set")

        eval_data = care.data
        self.assertEqual(len(eval_data), 1)
        self.assertEqual(eval_data.conversations[0].id, "1")
        self.assertEqual(eval_data.conversations[0].messages[0].contents[0].content, "test question")
        self.assertIn("dataset_version_id", eval_data.metadata)
        self.assertEqual(eval_data.metadata["dataset_version_id"], "0.0.0")


if __name__ == "__main__":
    unittest.main()
