import unittest
import uuid
import sys
from pathlib import Path
import logging
from unittest.mock import patch, MagicMock, Mock
from deepeval.metrics.hallucination.schema import (
    HallucinationScoreReason,
    HallucinationVerdict,
    Verdicts,
)

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.output.results import MetricsResults
from pre_deploy.query_processor import RequestDict
from pre_deploy.metrics.hallucination import (
    hallucination_batch_generate,
    start_hallucination,
    advance_hallucination,
    finalize_hallucination,
)
from tests.utils import create_eval_dataset_for_testing, assert_invalid_dataset_version_id_raises

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestHallucination(unittest.TestCase):
    def test_batch_hallucination_raises_for_invalid_dataset_version_id(self):
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")
        test_set = [
            {
                "input": "List campuses.",
                "actual_output": "There are 4 campuses in total: Campus A, Campus B, Campus C, Campus D.",
                "context": ["Campus A", "Campus B", "Campus C"]
            }
        ]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_set,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: hallucination_batch_generate(
                evaluator_info=evaluator_info,
                eval_dataset=eval_dataset,
                threshold=0.5,
                include_reason=True,
                strict_mode=False,
            ),
        )

    def test_batch_hallucination_basic_no_reference(self):
        """Test basic Hallucination generation with valid inputs."""
        evaluator_info = RequestDict(metric_name="test_metric", metric_phase="", run_id=str(uuid.uuid4()), username="test_user", model_name="gpt4o_mini", model_provider="openai")

        # Create sample test set
        test_set = [
            {
                "input": "List campuses.",
                "actual_output": "There are 4 campuses in total: Campus A, Campus B, Campus C, Campus D.",
                "context": ["Campus A", "Campus B", "Campus C"]
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(
            test_set,
            metadata={"dataset_version_id": "unit-test-v1"}
        )

        # Call the function
        results = hallucination_batch_generate(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            threshold=0.5,
            include_reason=True,
            strict_mode=False,
        )

        self.assertIsInstance(results, MetricsResults)
        if not isinstance(results, MetricsResults):
            self.fail("Expected MetricsResults object")

        self.assertEqual(len(results), 1, "Number of metrics should match number of test cases")

        results_json = results.to_dict()
        self.assertIsNotNone(results_json["run_id"], "Run ID should be populated.")

        for _, item in results_json["results"].items():
            self.assertIsNotNone(item.get('score'), "Score should be populated")
            self.assertIsInstance(item['score'], (int, float), "Score should be a number")
            self.assertIsNotNone(item.get('reason'), "Reason should be populated")
            self.assertIsInstance(item['reason'], str, "Reason should be a string")
            self.assertIsNotNone(item.get('success'), "Success should be populated")
            self.assertIsInstance(item['success'], bool, "Success should be a boolean")

    @patch('pre_deploy.metrics.hallucination.DeepEvalClient.enqueue_batch', autospec=True)
    def test_start_enqueues_verdicts_phase(self, mock_enqueue_batch):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1", "context": ["ctx"]}],
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

        result = start_hallucination(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "verdicts")
        self.assertTrue(mock_enqueue_batch.called)
        model_self = mock_enqueue_batch.call_args[0][0]
        self.assertEqual(model_self.request_dict.metric_phase, "verdicts")

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.hallucination.DeepEvalClient.enqueue_batch', autospec=True)
    @patch('pre_deploy.metrics.hallucination.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_to_reasons_finalizes_verdicts_then_enqueues(self, mock_finalize_batch, mock_enqueue_batch, mock_check_phase_status):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1", "context": ["ctx"]}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 1,
            "pending_items": 0,
            "is_complete": True,
        }
        mock_finalize_batch.return_value = {
            "0": Verdicts(verdicts=[HallucinationVerdict(verdict="yes", reason="ok")]),
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

        result = advance_hallucination(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            phase="reasons",
            include_reason=True,
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "reasons")
        self.assertTrue(mock_finalize_batch.called)
        self.assertTrue(mock_enqueue_batch.called)

    @patch('pre_deploy.metrics.utils.QueryProcessorClient.check_phase_status')
    @patch('pre_deploy.metrics.hallucination.DeepEvalClient.finalize_batch', autospec=True)
    def test_advance_returns_blocked_status_when_verdicts_incomplete(self, mock_finalize_batch, mock_check_phase_status):
        evaluator_info = RequestDict(
            metric_name="test_metric",
            metric_phase="",
            run_id=str(uuid.uuid4()),
            username="test_user",
            model_name="gpt4o_mini",
            model_provider="openai",
        )
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1", "context": ["ctx"]}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }
        mock_finalize_batch.return_value = {}

        result = advance_hallucination(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            phase="reasons",
            include_reason=True,
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result["is_complete"])
        self.assertEqual(result["completed_items"], 0)
        self.assertEqual(result["pending_items"], 1)

    @patch('pre_deploy.metrics.hallucination.DeepEvalClient.finalize_batch', autospec=True)
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
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1", "context": ["ctx"]}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 1,
            "pending_items": 0,
            "is_complete": True,
        }
        mock_finalize_batch.side_effect = [
            {"0": Verdicts(verdicts=[HallucinationVerdict(verdict="yes", reason="ok")])},
            {"0": HallucinationScoreReason(reason="overall reason")},
        ]

        results = finalize_hallucination(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
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
        eval_dataset = create_eval_dataset_for_testing(
            [{"input": "Q1", "actual_output": "A1", "context": ["ctx"]}],
            metadata={"dataset_version_id": "unit-test-v1"},
        )

        mock_check_phase_status.return_value = {
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = finalize_hallucination(
            evaluator_info=evaluator_info,
            eval_dataset=eval_dataset,
            include_reason=True,
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
