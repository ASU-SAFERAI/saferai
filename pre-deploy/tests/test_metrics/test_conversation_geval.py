import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock, Mock

from deepeval.metrics.g_eval.utils import Rubric
from deepeval.metrics.conversational_g_eval.schema import ReasonScore, Steps

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.conversation_geval import (
    conversation_geval_batch_generate,
    start_conversation_geval,
    advance_conversation_geval,
    finalize_conversation_geval,
)
from pre_deploy.output.results import MetricsResults
from tests.utils import (
    create_eval_dataset_for_testing,
    create_eval_conversation_dataset_for_testing,
    assert_invalid_dataset_version_id_raises,
)

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestConversationGEval(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        """Test GEval errors when dataset_version_id is missing or invalid."""
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_data = [{"input": "Hello!", "actual_output": "Hi!"}]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_data,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: conversation_geval_batch_generate(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                name="Relevance",
                evaluation_steps=["The answer is relevant."],
                threshold=0.5,
                strict_mode=False,
            ),
        )

    def test_batch_generate_basic_geval(self):
        """Test basic GEval generation with valid inputs."""
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(
            test_data,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        results = conversation_geval_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            evaluation_steps=[
                "The answer is relevant to the user's query.",
                "The answer is coherent and contextually appropriate."
            ],
            threshold=0.5,
            strict_mode=False
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

        # Test metric-specific attributes for G-Eval
        for _, metric in results.metrics.items():
            evaluation_steps = getattr(metric, 'evaluation_steps', None)
            self.assertIsNotNone(evaluation_steps, "Evaluation steps should be populated")
            self.assertIsInstance(evaluation_steps, list, "Evaluation steps should be a list")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_evaluation_response_parsing_error(self, mock_batch_generate):
        """Test that evaluation response parsing errors result in score 0 and 'error_parsing_response'."""
        # Mock batch_generate: first call valid for steps, second call invalid ({})
        mock_batch_generate.side_effect = [
            {"0": MagicMock(steps=["Step 1", "Step 2"])},  # Valid for steps
            {"0": {}}  # Invalid for eval response
        ]

        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi!",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = conversation_geval_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Test GEval",
            evaluation_steps=["Step 1", "Step 2"],
            threshold=0.5
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results.metrics), 1)
        score = getattr(results.metrics['0'], 'score', None)
        reason = getattr(results.metrics['0'], 'reason', None)
        success = getattr(results.metrics['0'], 'success', None)

        if score is not None:
            self.assertEqual(score, 0, "Score should be 0 on response parsing error")
        if reason is not None:
            self.assertEqual(reason, "error_parsing_response", "Reason should be 'error_parsing_response' on parsing error")
        if success is not None:
            self.assertFalse(success, "Success should be False on response parsing error")

    def test_geval_with_rubrics(self):
        """Test GEval generation with a list of Rubric objects."""
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Define rubrics for scoring guidance
        rubrics = [
            {"score_range": (0, 2), "expected_outcome": "Response is completely irrelevant or inappropriate"},
            {"score_range": (3, 5), "expected_outcome": "Response has some relevance but lacks depth or accuracy"},
            {"score_range": (6, 8), "expected_outcome": "Response is relevant and mostly accurate with minor issues"},
            {"score_range": (9, 10), "expected_outcome": "Response is highly relevant, accurate, and comprehensive"}
        ]

        test_data = [
            {
                "input": "What is the capital of France?",
                "actual_output": "The capital of France is Paris.",
            },
            {
                "input": "Can you tell me about its history?",
                "actual_output": "Paris has a rich history dating back over 2,000 years. It was originally a Celtic settlement called Lutetia, later becoming a Roman city. The city became the capital of France in 508 AD under Clovis I."
            },
            {
                "input": "How do I bake a cake?",
                "actual_output": "To bake a cake, you'll need ingredients like flour, sugar, eggs, and butter. Mix them together, pour into a pan, and bake at 350°F for about 30 minutes.",
            }
        ]
        eval_dataset = create_eval_conversation_dataset_for_testing(
            test_data,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        results = conversation_geval_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Relevance with Rubrics",
            evaluation_steps=[
                "Evaluate how relevant and accurate the assistant's responses are to the user's queries."
            ],
            threshold=0.6,
            rubrics=rubrics
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results), len(eval_dataset.conversations), "Number of metrics should match number of conversations")

        # Test through MetricsResults interface
        results_dict = results.to_dict()
        self.assertIn("name", results_dict)
        self.assertIn("results", results_dict)

        # Verify that all conversations have results in the dictionary
        for convo in eval_dataset.conversations:
            self.assertIn(convo.id, results_dict["results"], f"Conversation {convo.id} should have results")
            convo_result = results_dict["results"][convo.id]
            # Verify basic metric properties through the results dictionary
            if convo_result["score"] is not None:
                self.assertIsInstance(convo_result["score"], (int, float), "Score should be a number")
                self.assertGreaterEqual(convo_result["score"], 0.0, "Score should be non-negative")
                self.assertLessEqual(convo_result["score"], 1.0, "Score should not exceed 1.0")

            if convo_result["reason"] is not None:
                self.assertIsInstance(convo_result["reason"], str, "Reason should be a string")

            self.assertIsInstance(convo_result["success"], bool, "Success should be a boolean")

        # Test metric-specific attributes for rubrics
        for _, metric in results.metrics.items():
            # Verify rubrics are properly attached to the metric
            rubric = getattr(metric, 'rubric', None)
            self.assertIsNotNone(rubric, "Rubric should be populated")
            if rubric is not None:
                self.assertIsInstance(rubric, list, "Rubric should be a list")
                self.assertEqual(len(rubric), 4, "Should have 4 rubric entries")

                # Verify rubric structure
                for rubric_item in rubric:
                    self.assertIsInstance(rubric_item, Rubric, "Each rubric should be a Rubric instance")
                    self.assertIsInstance(rubric_item.score_range, tuple, "Score range should be a tuple")
                    self.assertEqual(len(rubric_item.score_range), 2, "Score range should have exactly 2 elements")
                    self.assertIsInstance(rubric_item.expected_outcome, str, "Expected outcome should be a string")
                    self.assertGreater(len(rubric_item.expected_outcome), 0, "Expected outcome should not be empty")

            # Verify that verbose logs include rubric information when available
            verbose_logs = getattr(metric, 'verbose_logs', None)
            if verbose_logs:
                self.assertIsInstance(verbose_logs, str, "Verbose logs should be a string")
                # The verbose logs should contain rubric information
                self.assertIn("Rubric:", verbose_logs, "Verbose logs should include rubric information")

    @patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_reason_score_when_evaluation_steps_are_provided(self, mock_enqueue_batch):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_conversation_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_enqueue_batch.return_value = {
            "run_id": evaluator_info.run_id,
            "metric_phase": "reason_score",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = start_conversation_geval(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            evaluation_steps=["The answer is relevant."],
        )

        self.assertEqual(result["metric_phase"], "reason_score")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "reason_score")

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.enqueue_batch', autospec=True)
    @patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_to_reason_score_enqueues_when_evaluation_steps_are_predefined(self, mock_finalize_batch, mock_enqueue_batch, mock_check_phase_status):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_conversation_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 1,
            "pending_items": 0,
            "is_complete": True,
        }
        mock_finalize_batch.return_value = {"0": Steps(steps=["s1"])}
        mock_enqueue_batch.return_value = {
            "run_id": evaluator_info.run_id,
            "metric_phase": "reason_score",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = advance_conversation_geval(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            phase="reason_score",
            name="Relevance",
            evaluation_steps=["The answer is relevant."],
        )

        self.assertEqual(result["metric_phase"], "reason_score")
        self.assertFalse(mock_finalize_batch.called)
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_returns_blocked_status_when_evaluation_steps_incomplete(self, mock_finalize_batch, mock_check_phase_status):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_conversation_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_enqueue_batch = MagicMock(return_value={
            "run_id": evaluator_info.run_id,
            "metric_phase": "reason_score",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        })

        with patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.enqueue_batch', mock_enqueue_batch):
            result = advance_conversation_geval(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                phase="reason_score",
                name="Relevance",
                evaluation_steps=["The answer is relevant."],
            )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["metric_phase"], "reason_score")
        self.assertEqual(result["pending_items"], 1)
        self.assertFalse(mock_finalize_batch.called)

    @patch('pre_deploy.metrics.conversation_geval.DeepEvalClient.finalize_batch', autospec=True)
    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    def test_finalize_reads_all_phases_and_returns_results(self, mock_check_phase_status, mock_finalize_batch):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_conversation_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 1,
            "pending_items": 0,
            "is_complete": True,
        }
        mock_finalize_batch.side_effect = [
            {"0": Steps(steps=["s1"])},
            {"0": ReasonScore(score=8, reason="The answer is relevant.")},
        ]

        results = finalize_conversation_geval(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            evaluation_steps=["The answer is relevant."],
            verbose_mode=False,
        )

        self.assertIsInstance(results, MetricsResults)
        self.assertEqual(mock_finalize_batch.call_count, 1)

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    def test_finalize_returns_incomplete_status_when_phase_incomplete(self, mock_check_phase_status):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_conversation_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = finalize_conversation_geval(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            evaluation_steps=["The answer is relevant."],
            verbose_mode=False,
        )

        self.assertIsInstance(result, dict)
        if isinstance(result, dict):
            self.assertFalse(result["is_complete"])
            self.assertEqual(result["pending_items"], 1)
            self.assertEqual(result["completed_items"], 0)


if __name__ == '__main__':
    unittest.main()
