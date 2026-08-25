import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch

from deepeval.metrics.answer_relevancy.schema import (
    AnswerRelevancyScoreReason,
    AnswerRelevancyVerdict,
    Statements,
    Verdicts,
)

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.answer_relevancy import (
    answer_relevancy_batch_generate,
    start_answer_relevancy,
    advance_answer_relevancy,
    finalize_answer_relevancy,
)
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_dataset_for_testing, assert_invalid_dataset_version_id_raises

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestAnswerRelevancy(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_set,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: answer_relevancy_batch_generate(
                evaluator_info=eval_run_info,
                eval_dataset=eval_dataset,
                verbose_mode=False,
            ),
        )

    def test_batch_generate_basic(self):
        # Build run info
        eval_run_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "Hello!",
                "actual_output": "Hi! How can I help?",
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(
            test_set,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        # Verify dataset metadata has required field for answer_relevancy
        self.assertIn("dataset_version_id", eval_dataset.metadata)
        self.assertEqual(eval_dataset.metadata["dataset_version_id"], "unit-test-v1")

        # Call the function
        results = answer_relevancy_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            verbose_mode=False
        )

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults when verbose_mode=False")

        results_json = results.to_dict()

        # Verify dataset_version_id is tracked in MetricsResults dataset info
        self.assertIn("dataset", results_json)
        self.assertIn("dataset_version_id", results_json["dataset"])
        self.assertEqual(results_json["dataset"]["dataset_version_id"], "unit-test-v1")

        # Intermediate runs are no longer produced; ensure top-level structure is correct
        self.assertIn("dataset", results_json)
        self.assertIn("run_id", results_json)
        self.assertIn("results", results_json)

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_statements_phase(self, mock_enqueue_batch):
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
            "metric_phase": "statements",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = start_answer_relevancy(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
        )

        self.assertEqual(result["metric_phase"], "statements")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "statements")

    @patch('pre_deploy.metrics.answer_relevancy.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.enqueue_batch', autospec=True)
    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_to_verdicts_finalizes_statements_then_enqueues(self, mock_finalize_batch, mock_enqueue_batch, mock_check_phase_status):
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
        mock_finalize_batch.return_value = {"0": Statements(statements=["s1"]) }
        mock_enqueue_batch.return_value = {
            "run_id": eval_run_info.run_id,
            "metric_phase": "verdicts",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
            "query_keys": ["0"],
        }

        result = advance_answer_relevancy(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            phase="verdicts",
        )

        self.assertEqual(result["metric_phase"], "verdicts")
        self.assertTrue(mock_finalize_batch.called)
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.answer_relevancy.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_returns_blocked_status_when_statements_incomplete(self, mock_finalize_batch, mock_check_phase_status):
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
        mock_finalize_batch.return_value = {}

        result = advance_answer_relevancy(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            phase="verdicts",
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["completed_items"], 0)
        self.assertEqual(result["pending_items"], 1)

    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.finalize_batch', autospec=True)
    @patch('pre_deploy.metrics.answer_relevancy.QueryProcessorClient.check_phase_status')
    def test_finalize_reads_all_phases_and_returns_results(self, mock_check_phase_status, mock_finalize_batch):
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
        mock_finalize_batch.side_effect = [
            {"0": Statements(statements=["s1"])},
            {"0": Verdicts(verdicts=[AnswerRelevancyVerdict(verdict="yes", reason="ok")])},
            {"0": AnswerRelevancyScoreReason(reason="overall reason")},
        ]

        results = finalize_answer_relevancy(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            verbose_mode=False,
        )

        self.assertIsInstance(results, MetricsResults)
        self.assertEqual(mock_finalize_batch.call_count, 3)

    @patch('pre_deploy.metrics.answer_relevancy.QueryProcessorClient.check_phase_status')
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

        result = finalize_answer_relevancy(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            verbose_mode=False,
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("is_complete"))
        self.assertEqual(result.get("pending_items"), 1)
        self.assertEqual(result.get("completed_items"), 0)

    @patch('pre_deploy.metrics.answer_relevancy.DeepEvalClient.batch_generate', autospec=True)
    def test_batch_generate_reason_prompts_use_per_index_input(self, mock_batch_generate):
        eval_run_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [
                {"input": "Q1", "actual_output": "A1"},
                {"input": "Q2", "actual_output": "A2"},
            ],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_batch_generate.side_effect = [
            {
                "0": Statements(statements=["s1"]),
                "1": Statements(statements=["s2"]),
            },
            {
                "0": Verdicts(verdicts=[AnswerRelevancyVerdict(verdict="yes", reason="ok0")]),
                "1": Verdicts(verdicts=[AnswerRelevancyVerdict(verdict="no", reason="ok1")]),
            },
            {
                "0": AnswerRelevancyScoreReason(reason="r0"),
                "1": AnswerRelevancyScoreReason(reason="r1"),
            },
        ]

        answer_relevancy_batch_generate(
            evaluator_info=eval_run_info,
            eval_dataset=eval_dataset,
            verbose_mode=False,
        )

        third_call_kwargs = mock_batch_generate.call_args_list[2].kwargs
        reasons_prompts = third_call_kwargs["prompts"]
        self.assertIn("Q1", reasons_prompts["0"])
        self.assertIn("Q2", reasons_prompts["1"])


if __name__ == '__main__':
    unittest.main()
