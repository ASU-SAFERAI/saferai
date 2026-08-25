import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict, AWSEnvironment
from pre_deploy.metrics.conversation_completeness import conversation_completeness_batch_generate
from pre_deploy.metrics.utils import _prompt_key
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_conversation_dataset_for_testing, create_eval_dataset_for_testing

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestCrossAccountMetric(unittest.TestCase):
    def test_batch_generate_basic_completion_cross_account(self):
        # Create a DeepEvalClient instance
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test data for EvalDataset
        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I assist you today?",
            },
            {
                "input": "What's the weather like today?",
                "actual_output": "It's sunny and warm outside.",
            }
        ]

        # Create EvalDataset from test data
        eval_dataset = create_eval_conversation_dataset_for_testing(test_data)

        # Create environment with different target account ID and role
        environment = AWSEnvironment(target_account_id="YOUR_TARGET_ACCOUNT_ID", role_name="YOUR_ROLE_NAME")

        # Call the function with default parameters and cross-account environment
        results = conversation_completeness_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True,
            environment=environment
        )

        # Assertions for basic completion
        self.assertIsInstance(results, MetricsResults, "Should return MetricsResults object")
        self.assertEqual(len(results), len(eval_dataset.conversations), "Number of metrics should match number of conversations")

        # Test the to_dict() method returns expected structure
        results_dict = results.to_dict()
        self.assertIn("name", results_dict)
        self.assertIn("dataset", results_dict)
        self.assertIn("results", results_dict)
        self.assertEqual(len(results_dict["results"]), len(eval_dataset.conversations))

        # Verify that all conversations have results in the dictionary
        for convo in eval_dataset.conversations:
            self.assertIn(convo.id, results_dict["results"], f"Conversation {convo.id} should have results")
            convo_result = results_dict["results"][convo.id]
            self.assertIn("score", convo_result, "Result should have score")
            self.assertIn("reason", convo_result, "Result should have reason")
            self.assertIn("success", convo_result, "Result should have success")
            self.assertIsInstance(convo_result["score"], (int, float), "Score should be a number")
            self.assertIsInstance(convo_result["reason"], (str, type(None)), "Reason should be string or None")
            self.assertIsInstance(convo_result["success"], bool, "Success should be a boolean")

        # Test that each individual metric still has the expected attributes for conversation completeness
        for _, metric in results.metrics.items():
            # Use getattr to safely access metric-specific attributes
            user_intentions = getattr(metric, 'user_intentions', None)
            self.assertIsNotNone(user_intentions, "User intentions should be populated")
            self.assertIsInstance(user_intentions, list, "User intentions should be a list")
            if user_intentions:
                self.assertGreater(len(user_intentions), 0, "User intentions should not be empty")

            verdicts = getattr(metric, 'verdicts', None)
            self.assertIsNotNone(verdicts, "Verdicts should be populated")
            self.assertIsInstance(verdicts, list, "Verdicts should be a list")


if __name__ == '__main__':
    unittest.main()
