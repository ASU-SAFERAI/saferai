"""
Test module for DAG-based fact checking with GEval verdict nodes.

This module contains tests specifically for the fact checking DAG implementation
that uses GEval nodes as verdict nodes in addition to traditional binary and
non-binary judgment nodes.
"""

import unittest
import sys
from pathlib import Path
import logging
from unittest.mock import patch
import uuid

from deepeval.test_case import LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.nodes import (
    TaskNode,
    BinaryJudgementNode,
    VerdictNode,
)

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())
from pre_deploy.query_processor import DeepEvalClient, RequestDict, AWSEnvironment
from pre_deploy.metrics.dag import dag_batch_generate
from pre_deploy.output.results import MetricsResults
from tests.utils import create_eval_dataset_for_testing, assert_invalid_dataset_version_id_raises

logging.getLogger('pre_deploy').setLevel(logging.INFO)

model = DeepEvalClient(
    RequestDict(username="test_user", metric_name="dag_metric", metric_phase="",
                model_name="gpt4o_mini", model_provider="openai", run_id=str(uuid.uuid4())),
    environment=AWSEnvironment(target_account_id=None, role_name=None)
)

# DAG definition for fact checking with GEval verdict nodes
# This DAG extracts facts and assertions, then uses both binary judgment
# and GEval nodes to assess the accuracy of the facts
fact_judgment_node = GEval(
    name="fact_check",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    evaluation_steps=[
        "Are all the facts and assertions listed in `facts` (previous nodes) accurate?"
    ],
    model=model
)

fact_detection_node = BinaryJudgementNode(
    criteria="Are all assertions in `facts` accurate?",
    children=[
        VerdictNode(verdict=True, score=10),
        VerdictNode(verdict=False, child=fact_judgment_node)
    ]
)

extract_facts_node = TaskNode(
    instructions="Extract all facts and assertions stated in `actual_output` in response to `input` in the "
                    "format of a numbered list.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="facts",
    children=[fact_detection_node]
)

fact_checker_dag = DeepAcyclicGraph(root_nodes=[extract_facts_node])


class TestFactCheckDAG(unittest.TestCase):
    """Test class for fact checking DAG with GEval verdict nodes."""

    def test_fact_checker_dag_raises_for_invalid_dataset_version_id(self):
        test_data = [
            {
                "input": "What is 2+2?",
                "actual_output": "2+2 equals 4.",
                "expected_output": "2+2=4"
            }
        ]

        assert_invalid_dataset_version_id_raises(
            test_case=self,
            test_set=test_data,
            create_dataset=create_eval_dataset_for_testing,
            run_metric_with_dataset=lambda eval_dataset: dag_batch_generate(
                model=model,
                eval_dataset=eval_dataset,
                name="Fact Checker",
                dag=fact_checker_dag,
                threshold=0.5,
                strict_mode=False,
            ),
        )

    def test_fact_checker_dag_basic_execution(self):
        """Test basic fact checker DAG execution with various accuracy levels."""
        test_data = [
            # Accurate facts
            {
                "input": "What is the capital of France?",
                "actual_output": "The capital of France is Paris. It is located in Europe.",
                "expected_output": "Paris is the capital of France."
            },
            # Mixed accuracy (should trigger GEval)
            {
                "input": "Tell me about the Earth.",
                "actual_output": "Earth is the third planet from the Sun. It is flat and supported by elephants. Earth has one moon.",
                "expected_output": "Facts about Earth."
            },
            # Inaccurate facts (should trigger GEval)
            {
                "input": "What is the largest planet?",
                "actual_output": "Jupiter is the largest planet. It is closer to the Sun than Earth.",  # Second fact is wrong
                "expected_output": "Jupiter is the largest planet."
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = dag_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            name="Fact Checker",
            dag=fact_checker_dag,
            threshold=0.5,
            strict_mode=False
        )

        self.assertIsInstance(results, MetricsResults, "Should return MetricsResults object")
        self.assertEqual(len(results), len(eval_dataset.conversations), "Number of metrics should match number of conversations")

        # Test the to_dict() method returns expected structure
        results_dict = results.to_dict()
        print(results_dict)
        self.assertIn("name", results_dict)
        self.assertIn("dataset", results_dict)
        self.assertIn("run_id", results_dict)
        self.assertIn("results", results_dict)

        # Verify that all conversations have results in the dictionary
        for i, convo in enumerate(eval_dataset.conversations):
            with self.subTest(test_case=i):
                self.assertIn(convo.id, results_dict["results"], f"Conversation {convo.id} should have results")
                convo_result = results_dict["results"][convo.id]
                self.assertIn("score", convo_result, "Result should have score")
                self.assertIn("reason", convo_result, "Result should have reason")
                self.assertIn("success", convo_result, "Result should have success")
                self.assertIsInstance(convo_result["score"], (int, float, type(None)), f"Score should be a number or None for test case {i}")
                self.assertIsInstance(convo_result["reason"], (str, type(None)), f"Reason should be string or None for test case {i}")
                self.assertIsInstance(convo_result["success"], bool, f"Success should be a boolean for test case {i}")

    @patch('pre_deploy.query_processor.DeepEvalClient.batch_generate')
    def test_fact_checker_error_handling(self, mock_batch_generate):
        """Test error handling when GEval nodes fail."""
        # Mock should return a dictionary with index as key, not a list
        mock_batch_generate.return_value = {0: {}}  # Changed from [{}] to {0: {}}

        test_data = [
            {
                "input": "What is 2+2?",
                "actual_output": "2+2 equals 4.",
                "expected_output": "2+2=4"
            }
        ]
        eval_dataset = create_eval_dataset_for_testing(test_data, metadata={"dataset_version_id": "unit-test-v1"})

        results = dag_batch_generate(
            model=model,
            eval_dataset=eval_dataset,
            name="Fact Checker with Error",
            dag=fact_checker_dag,
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
            self.assertIn("score", convo_result, "Score should be populated even on error")
            self.assertIn("reason", convo_result, "Reason should be populated even on error")


if __name__ == '__main__':
    unittest.main()
