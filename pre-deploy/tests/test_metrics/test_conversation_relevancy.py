import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock, Mock

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.conversation_relevancy import (
    conversation_relevancy_batch_generate,
    start_conversation_relevancy,
    advance_conversation_relevancy,
    finalize_conversation_relevancy,
)
from pre_deploy.metrics.utils import _prompt_key
from pre_deploy.output.results import MetricsResults
from tests.utils import (
    create_eval_dataset_for_testing,
    create_eval_conversation_dataset_for_testing,
    assert_invalid_dataset_version_id_raises,
)

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestConversationRelevancy(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_data = [{"input": "Hello!", "actual_output": "Hi!"}]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_data,
            create_dataset=create_eval_conversation_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: conversation_relevancy_batch_generate(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                threshold=0.5,
                include_reason=True,
            ),
        )

    def test_batch_generate_basic_relevancy(self):
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

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
        eval_dataset = create_eval_conversation_dataset_for_testing(
            test_data,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        results = conversation_relevancy_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        # Assertions for basic relevancy
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
            self.assertIsInstance(convo_result["score"], (int, float, type(None)), "Score should be a number or None")
            self.assertIsInstance(convo_result["reason"], (str, type(None)), "Reason should be string or None")
            self.assertIsInstance(convo_result["success"], bool, "Success should be a boolean")

        # Test metric-specific attributes for conversation relevancy
        for _, metric in results.metrics.items():
            verdicts = getattr(metric, 'verdicts', None)
            self.assertIsNotNone(verdicts, "Verdicts should be populated")
            if verdicts is not None:
                self.assertIsInstance(verdicts, list, "Verdicts should be a list")
                self.assertGreater(len(verdicts), 0, "Verdicts should not be empty")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_verdict_parsing_error(self, mock_batch_generate):
        """Test that verdict parsing errors result in 'error_parsing_verdict'."""
        # Mock batch_generate: first call invalid ({}), second call valid for reason if include_reason=True
        mock_batch_generate.side_effect = [
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

        results = conversation_relevancy_batch_generate(
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
        if verdicts and len(verdicts) > 0:
            self.assertEqual(verdicts[0].verdict, "not_relevant")
            self.assertEqual(verdicts[0].reason, "error_parsing_verdict")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_reason_parsing_error(self, mock_batch_generate):
        """Test that reason parsing errors result in 'reason_not_found'."""
        # Mock batch_generate: first call valid for verdicts, second call invalid (missing 'reason')
        mock_batch_generate.side_effect = [
            {_prompt_key("0", 0): MagicMock(verdict="yes", reason="test")},  # Valid for verdicts
            {"0": {"invalid": "data"}}  # Invalid for reason
        ]

        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        test_data = [
            {
                "input": "Hello!",
                "actual_output": "Hi!",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = conversation_relevancy_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")
        self.assertEqual(len(results.metrics), 1)

        reason = getattr(results.metrics['0'], 'reason', None)
        if reason is not None:
            self.assertEqual(reason, "reason_not_found")

    @patch('pre_deploy.metrics.conversation_relevancy.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_verdicts_phase(self, mock_enqueue_batch):
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
            "metric_phase": "verdicts",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = start_conversation_relevancy(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "verdicts")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "verdicts")

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.conversation_relevancy.DeepEvalClient.enqueue_batch', autospec=True)
    @patch('pre_deploy.metrics.conversation_relevancy.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_to_reasons_finalizes_verdicts_then_enqueues(self, mock_finalize_batch, mock_enqueue_batch, mock_check_phase_status):
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
        mock_finalize_batch.return_value = {
            _prompt_key("0", 0): MagicMock(verdict="yes", reason="ok"),
        }
        mock_enqueue_batch.return_value = {
            "run_id": evaluator_info.run_id,
            "metric_phase": "reasons",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = advance_conversation_relevancy(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            phase="reasons",
            threshold=0.5,
            include_reason=True,
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "reasons")
        self.assertTrue(mock_finalize_batch.called)
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.conversation_relevancy.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_returns_blocked_status_when_verdicts_incomplete(self, mock_finalize_batch, mock_check_phase_status):
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
        mock_finalize_batch.return_value = {}

        result = advance_conversation_relevancy(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            phase="reasons",
            threshold=0.5,
            include_reason=True,
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["completed_items"], 0)
        self.assertEqual(result["pending_items"], 1)

    @patch('pre_deploy.metrics.conversation_relevancy.DeepEvalClient.finalize_batch', autospec=True)
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
            {_prompt_key("0", 0): MagicMock(verdict="yes", reason="ok")},
            {"0": MagicMock(reason="overall reason")},
        ]

        results = finalize_conversation_relevancy(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True,
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertIsInstance(results, MetricsResults)
        self.assertEqual(mock_finalize_batch.call_count, 2)

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

        result = finalize_conversation_relevancy(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True,
            verbose_mode=False,
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("is_complete"))
        self.assertEqual(result.get("pending_items"), 1)
        self.assertEqual(result.get("completed_items"), 0)


if __name__ == '__main__':
    unittest.main()
