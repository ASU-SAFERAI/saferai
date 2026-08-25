import unittest
import uuid
import sys
from pathlib import Path
import logging
from typing import Any, cast
from unittest.mock import patch, MagicMock, Mock
from deepeval.metrics.g_eval.schema import ReasonScore

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.custom import (
    custom_batch_generate,
    start_custom,
    advance_custom,
    finalize_custom,
)
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_dataset_for_testing, assert_invalid_dataset_version_id_raises

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestCustom(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_set = [{"input": "Hello!", "actual_output": "Hi!"}]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_set,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: custom_batch_generate(
                evaluator_info=eval_run_info,
                eval_dataset=eval_dataset,
                name="Relevance",
                with_reference=False,
                evaluation_steps=["The answer is relevant to the user's query."],
                threshold=0.5,
                strict_mode=False,
                verbose_mode=False,
                environment=Mock(),
            ),
        )

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate', autospec=True)
    def test_batch_generate_basic_custom_no_reference(self, mock_batch_generate):
        """Test basic GEval generation with valid inputs."""
        # Build run info
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        evaluation_steps=[
            "The answer is relevant to the user's query.",
            "The answer is coherent and contextually appropriate."
        ]
        mock_batch_generate.return_value = {
            "0": ReasonScore(score=8, reason="looks relevant"),
        }

        # Call the function
        results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            with_reference=False,
            evaluation_steps=evaluation_steps,
            threshold=0.5,
            strict_mode=False,
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")

        results_json = results.to_dict()
        self.assertIsNotNone(results_json["run_id"], "Run ID should be populated.")

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    def test_check_evaluation_steps_not_provided(self):
        # Build run info
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})

        # Assert that a ValueError is raised when evaluation_steps is empty
        with self.assertRaises(ValueError):
            results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Test GEval",
            with_reference=False,
            evaluation_steps=[],
            threshold=0.5,
            verbose_mode=False,
            environment=Mock(),
            )

        # Assert that a ValueError is raised when evaluation_steps is None
        with self.assertRaises(ValueError):
            results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Test GEval",
            with_reference=False,
            evaluation_steps=cast(Any, None),
            threshold=0.5,
            verbose_mode=False,
            environment=Mock(),
            )

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate', autospec=True)
    def test_different_evaluation_steps_for_different_data(self, mock_batch_generate):
        """Test GEval generation with dict evaluation steps as inputs."""
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            },
            {
                "input": "Hello!",
                "actual_output": "The weather is so nice.",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        evaluation_steps={
            "0": [
                "The answer is relevant to the user's query.",
            ],
            "1": [
                "The answer mentions something about weather."
            ]
        }
        mock_batch_generate.return_value = {
            "0": ReasonScore(score=8, reason="relevant"),
            "1": ReasonScore(score=7, reason="mentions weather"),
        }

        # Call the function
        results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            with_reference=False,
            evaluation_steps=evaluation_steps,
            threshold=0.5,
            strict_mode=False,
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertEqual(len(results), 2, "Number of metrics should match number of test cases")

        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")

        results_json = results.to_dict()

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate', autospec=True)
    def test_custom_with_rubrics(self, mock_batch_generate):
        """Test GEval generation with a list of Rubric objects."""
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Define rubrics for scoring guidance
        rubrics = [
            {
                "score_range": (0, 2),
                "expected_outcome": "Response is completely irrelevant or inappropriate"
            },
            {
                "score_range": (3, 5),
                "expected_outcome": "Response has some relevance but lacks depth or accuracy"
            },
            {
                "score_range": (6, 8),
                "expected_outcome": "Response is relevant and mostly accurate with minor issues"
            },
            {
                "score_range": (9, 10),
                "expected_outcome": "Response is highly relevant, accurate, and comprehensive"
            }
        ]

        test_set = [
            {
                "input": "What is the capital of France?",
                "actual_output": "The capital of France is Paris.",
            },
            {
                "input": "How do you bake a cake?",
                "actual_output": "To bake a cake, you'll need ingredients like flour, sugar, eggs, and butter. Mix them together, pour into a pan, and bake at 350°F for about 30 minutes.",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        mock_batch_generate.return_value = {
            "0": ReasonScore(score=9, reason="highly relevant"),
            "1": ReasonScore(score=8, reason="mostly relevant"),
        }

        results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Relevance with Rubrics",
            with_reference=False,
            evaluation_steps=["Evaluate how relevant and accurate the responses are to the input queries."],
            threshold=0.6,
            rubrics=rubrics,
            strict_mode=False,
            verbose_mode=False,
            environment=Mock(),
        )

        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        results_json = results.to_dict()

        self.assertEqual(len(results), len(test_set), "Number of metrics should match number of test cases")

        for _, item in results_json["results"].items():
            # Verify basic metric properties
            self.assertIsNotNone(item['score'], "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertGreaterEqual(item['score'], 0.0, "Score should be non-negative")
            self.assertLessEqual(item['score'], 1.0, "Score should not exceed 1.0")

            self.assertIsNotNone(item['reason'], "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")

            self.assertIsNotNone(item['success'], "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate', autospec=True)
    def test_custom_with_different_rubrics(self, mock_batch_generate):
        """Test GEval generation with a list of Rubric objects."""

        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Define rubrics for scoring guidance
        rubrics = {
            "0": [
                {
                    "score_range": (0, 5),
                    "expected_outcome": "Response is completely irrelevant or inappropriate"
                },
                {
                    "score_range": (6, 10),
                    "expected_outcome": "Response is highly relevant, accurate, and comprehensive"
                }
            ],
            "1": [
                {
                    "score_range": (0, 2),
                    "expected_outcome": "Response is completely irrelevant or inappropriate"
                },
                {
                    "score_range": (3, 10),
                    "expected_outcome": "Response is highly relevant, accurate, and comprehensive"
                }
            ]
        }

        test_set = [
            {
                "input": "What is the capital of France?",
                "actual_output": "The capital of France is Paris.",
            },
            {
                "input": "How do you bake a cake?",
                "actual_output": "To bake a cake, you'll need ingredients like flour, sugar, eggs, and butter. Mix them together, pour into a pan, and bake at 350°F for about 30 minutes.",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        mock_batch_generate.return_value = {
            "0": ReasonScore(score=9, reason="highly relevant"),
            "1": ReasonScore(score=8, reason="mostly relevant"),
        }

        results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Relevance with Rubrics",
            with_reference=False,
            evaluation_steps=["Evaluate how relevant and accurate the responses are to the input queries."],
            threshold=0.6,
            rubrics=rubrics,
            strict_mode=False,
            verbose_mode=False,
            environment=Mock(),
        )

        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        results_json = results.to_dict()

        self.assertEqual(len(results), len(test_set), "Number of metrics should match number of test cases")

        for _, item in results_json["results"].items():
            # Verify basic metric properties
            self.assertIsNotNone(item['score'], "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertGreaterEqual(item['score'], 0.0, "Score should be non-negative")
            self.assertLessEqual(item['score'], 1.0, "Score should not exceed 1.0")

            self.assertIsNotNone(item['reason'], "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")

            self.assertIsNotNone(item['success'], "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate', autospec=True)
    def test_batch_generate_verbose_output(self, mock_batch_generate):
        """Test basic GEval generation with valid inputs and verbose outputs."""
        # Build run info
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_set, metadata={"dataset_version_id": "unit-test-v1"})
        evaluation_steps=[
            "The answer is relevant to the user's query.",
            "The answer is coherent and contextually appropriate."
        ]
        mock_batch_generate.return_value = {
            "0": ReasonScore(score=8, reason="looks relevant"),
        }

        # Call the function
        results = custom_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            name="Relevance",
            with_reference=False,
            evaluation_steps=evaluation_steps,
            threshold=0.5,
            strict_mode=False,
            verbose_mode=True,
            environment=Mock(),
        )

        self.assertTrue(isinstance(results, dict), "Verbose output should be a dictionary")
        if not isinstance(results, dict):
            self.fail("Expected verbose output dictionary")
        self.assertIn("input", results['0'], "Verbose output should contain 'input'")
        self.assertIn("actual_output", results['0'], "Verbose output should contain 'actual_output'")
        self.assertIn("expected_output", results['0'], "Verbose output should contain 'expected_output'")
        self.assertIn("score", results['0'], "Verbose output should contain 'score'")
        self.assertIn("reason", results['0'], "Verbose output should contain 'reason'")
        self.assertIn("success", results['0'], "Verbose output should contain 'success'")

    @patch('pre_deploy.query_processor.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_reason_score_phase(self, mock_enqueue_batch):
        eval_run_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_enqueue_batch.return_value = {
            "run_id": eval_run_info.run_id,
            "metric_phase": "reason_score",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = start_custom(
            name="Relevance",
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            with_reference=False,
            evaluation_steps=["The answer is relevant."],
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "reason_score")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "reason_score")

    @patch('pre_deploy.query_processor.DeepEvalClient.enqueue_batch', autospec=True)
    def test_advance_reenqueues_reason_score_phase(self, mock_enqueue_batch):
        eval_run_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_enqueue_batch.return_value = {
            "run_id": eval_run_info.run_id,
            "metric_phase": "reason_score",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = advance_custom(
            name="Relevance",
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            phase="reason_score",
            with_reference=False,
            evaluation_steps=["The answer is relevant."],
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "reason_score")
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.query_processor.DeepEvalClient.finalize_batch', autospec=True)
    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    def test_finalize_reads_phase_and_returns_results(self, mock_check_phase_status, mock_finalize_batch):
        eval_run_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 1,
            "pending_items": 0,
            "is_complete": True,
        }
        mock_finalize_batch.return_value = {
            "0": ReasonScore(score=8, reason="overall reason"),
        }

        results = finalize_custom(
            name="Relevance",
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            with_reference=False,
            evaluation_steps=["The answer is relevant."],
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertIsInstance(results, MetricsResults)
        self.assertEqual(mock_finalize_batch.call_count, 1)

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    def test_finalize_returns_incomplete_status_when_phase_incomplete(self, mock_check_phase_status):
        eval_run_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1"}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = finalize_custom(
            name="Relevance",
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            with_reference=False,
            evaluation_steps=["The answer is relevant."],
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        if isinstance(result, dict):
            self.assertFalse(result["is_complete"])
            self.assertEqual(result["pending_items"], 1)
            self.assertEqual(result["completed_items"], 0)


if __name__ == '__main__':
    unittest.main()
