import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.conversation_completeness import conversation_completeness_batch_generate
from pre_deploy.metrics.utils import _prompt_key
from pre_deploy.output.results import MetricsResults
from tests.utils import (
    create_eval_conversation_dataset_for_testing,
    create_eval_dataset_for_testing,
    assert_invalid_dataset_version_id_raises,
)

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestConversationCompleteness(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_data = [{"input": "Hello!", "actual_output": "Hi!"}]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_data,
            create_dataset=create_eval_conversation_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: conversation_completeness_batch_generate(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                threshold=0.5,
                include_reason=True,
            ),
        )

    def test_batch_generate_basic_completion(self):
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
        eval_dataset = create_eval_conversation_dataset_for_testing(
            test_data,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        # Call the function with default parameters
        results = conversation_completeness_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        # Assertions for basic completion
        self.assertIsInstance(results, MetricsResults, "Should return MetricsResults object")
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results), len(eval_dataset.conversations), "Number of metrics should match number of conversations")

        # Test the to_dict() method returns expected structure
        results_dict = results.to_dict()
        self.assertIn("name", results_dict)
        self.assertIn("dataset", results_dict)
        self.assertIsNotNone(results_dict["run_id"], "Run ID should be populated.")
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

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_user_intentions_parsing_error(self, mock_batch_generate):
        """Test that UserIntentions parsing errors default to ['error_parsing_intentions']."""
        # Mock batch_generate: first call returns invalid ({}), others return valid defaults
        mock_batch_generate.side_effect = [
            {"0": {}},  # Invalid for user intentions
            {_prompt_key("0", 0): MagicMock(verdict="yes", reason="test")},  # Valid for verdicts
            {"0": {"reason": "test_reason"}}  # Valid for reason
        ]

        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi!",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = conversation_completeness_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results.metrics), 1)
        user_intentions = getattr(results.metrics['0'], 'user_intentions', None)
        self.assertEqual(user_intentions, ["error_parsing_intentions"])

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_verdict_parsing_error(self, mock_batch_generate):
        """Test that verdict parsing errors result in 'error_parsing_verdict'."""
        # Mock batch_generate: first call valid, second call invalid ({}), third call valid
        mock_batch_generate.side_effect = [
            {"0": MagicMock(intentions=["test_intention"])},  # Valid for user intentions
            {_prompt_key("0", 0): {}},  # Invalid for verdicts
            {"0": {"reason": "test_reason"}}  # Valid for reason
        ]

        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi!",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = conversation_completeness_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results.metrics), 1)
        verdicts = getattr(results.metrics['0'], 'verdicts', None)
        self.assertIsNotNone(verdicts)
        if verdicts:  # Type narrowing
            self.assertEqual(len(verdicts), 1)
            self.assertEqual(verdicts[0].verdict, "no")
            self.assertEqual(verdicts[0].reason, "error_parsing_verdict")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_reason_parsing_error(self, mock_batch_generate):
        """Test that reason parsing errors result in 'reason_not_found'."""
        # Mock batch_generate: first two calls valid, third call invalid (missing 'reason')
        mock_batch_generate.side_effect = [
            {"0": MagicMock(intentions=["test_intention"])},  # Valid for user intentions
            {_prompt_key("0", 0): MagicMock(verdict="yes", reason="test")},  # Valid for verdicts
            {"0": {"invalid": "data"}}  # Invalid for reason (no 'reason' key)
        ]

        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi!",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = conversation_completeness_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results.metrics), 1)
        self.assertEqual(results.metrics['0'].reason, "reason_not_found")


if __name__ == '__main__':
    unittest.main()
