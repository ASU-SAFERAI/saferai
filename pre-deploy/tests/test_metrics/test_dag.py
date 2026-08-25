"""
Test module for DAG-based evaluation metrics focusing on summary structure validation.

This module contains tests for the basic DAG implementation that evaluates
document structure using traditional binary and non-binary judgment nodes.
For tests involving GEval verdict nodes and fact checking, see test_fact_check_dag.py.
"""

import unittest
import sys
from pathlib import Path
import logging
from unittest.mock import patch
import uuid

from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.nodes import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import DeepEvalClient, RequestDict, AWSEnvironment
from pre_deploy.metrics.dag import dag_batch_generate
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_dataset_for_testing, assert_invalid_dataset_version_id_raises

logging.getLogger('pre_deploy').setLevel(logging.INFO)

model = DeepEvalClient(
    request_dict=RequestDict(username="test_user", metric_name="dag_metric", metric_phase="",
                             model_name="gpt4o_mini", model_provider="openai", run_id=str(uuid.uuid4())),
    environment=AWSEnvironment(target_account_id=None, role_name=None)
)

# DAG definition from test_dag.py for testing summary structure
correct_order_node = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)

correct_headings_node = BinaryJudgementNode(
    criteria="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=correct_order_node),
    ],
)

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
    children=[correct_headings_node, correct_order_node],
)

# Create the DAG
dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])


class TestDAGBatchGenerate(unittest.TestCase):
    def test_batch_generate_raises_for_invalid_dataset_version_id(self):
        test_data = [
            {
                "input": "Summarize the article.",
                "actual_output": "Intro: This is intro. Body: This is body. Conclusion: This is conclusion.",
                "expected_output": "A good summary."
            }
        ]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_data,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: dag_batch_generate(
                model=model,
                eval_dataset=eval_dataset,
                name="Summary Quality",
                dag=dag,
                threshold=0.5,
                strict_mode=False,
            ),
        )

    def test_batch_generate_basic_dag(self):
        """Test basic DAG batch generation with valid inputs."""
        test_data = [
            {
                "input": "Summarize the article.",
                "actual_output": "Intro: This is intro. Body: This is body. Conclusion: This is conclusion.",
                "expected_output": "A good summary."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = dag_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            name="Summary Quality",
            dag=dag,
            threshold=0.5,
            strict_mode=False
        )

        self.assertIsInstance(results, MetricsResults, "Should return MetricsResults object")
        self.assertEqual(len(results), len(eval_dataset.conversations), "Number of metrics should match number of conversations")

        # Test the to_dict() method returns expected structure
        results_dict = results.to_dict()
        self.assertIn("name", results_dict)
        self.assertIn("dataset", results_dict)
        self.assertIn("run_id", results_dict)
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
            print(convo_result["score"], convo_result["reason"], convo_result["success"])

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_batch_generate_parsing_error(self, mock_batch_generate):
        """Test that parsing errors in batch_generate result in appropriate defaults."""
        # Mock batch_generate to return invalid responses
        mock_batch_generate.return_value = [{}]  # Invalid dict for parsing

        test_data = [
            {
                "input": "Summarize the article.",
                "actual_output": "Intro: This is intro. Body: This is body. Conclusion: This is conclusion.",
                "expected_output": "A good summary."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = dag_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            name="Summary Quality",
            dag=dag,
            threshold=0.5,
            strict_mode=False
        )

        self.assertIsInstance(results, MetricsResults)
        self.assertEqual(len(results), 1)

        # Test through MetricsResults interface
        results_dict = results.to_dict()
        self.assertIn("name", results_dict)
        self.assertIn("results", results_dict)

        # Verify each conversation has results
        for convo in eval_dataset.conversations:
            self.assertIn(convo.id, results_dict["results"])
            convo_result = results_dict["results"][convo.id]
            # Assuming parsing error leads to score 0 or default; adjust based on actual behavior
            if convo_result["score"] is not None:
                self.assertEqual(convo_result["score"], 0, "Score should be 0 on parsing error")
            if convo_result["reason"] and str(convo_result["reason"]).lower():
                self.assertIn("error", str(convo_result["reason"]).lower(), "Reason should indicate error on parsing failure")
            self.assertFalse(convo_result["success"], "Success should be False on parsing error")

if __name__ == '__main__':
    unittest.main()
