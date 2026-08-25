"""This is a minimal batched implementation of _execute() for VerdictNode,
which is the terminal node in the DAG."""
from typing import Dict, List, Optional, Sequence

from deepeval.metrics.dag.nodes import (
    VerdictNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    decrement_indegree,
    construct_node_verbose_log
)
from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics.dag.schema import (
    MetricScoreReason,
)
from deepeval.metrics.dag.templates import (
    VerdictNodeTemplate,
)

from ...query_processor import DeepEvalClient
from ..custom import custom_batch_generate
from ...loaders.deepeval_test_cases import deepeval_llm_test_cases_to_dataset


def verdict_node_batch_generate(model: DeepEvalClient, nodes: List[VerdictNode],
                                metrics: Sequence[BaseMetric], test_cases: List[LLMTestCase],
                                depth: int) -> None:
    # Filter valid nodes first
    valid_nodes = []
    valid_metrics = []
    valid_test_cases = []

    for node, metric, test_case in zip(nodes, metrics, test_cases):
        decrement_indegree(node)
        if node._indegree > 0:
            continue

        is_nonbinary_judgment_node = isinstance(node._parent, NonBinaryJudgementNode)
        is_binary_judgment_node = isinstance(node._parent, BinaryJudgementNode)

        if is_nonbinary_judgment_node or is_binary_judgment_node:
            if node._parent._verdict.verdict != node.verdict:
                continue

        # Add ALL valid nodes, not just those with children
        valid_nodes.append(node)
        valid_metrics.append(metric)
        valid_test_cases.append(test_case)

    # Separate nodes by execution type
    deterministic_nodes = []
    deterministic_metrics = []
    deterministic_test_cases = []

    geval_nodes = []
    geval_metrics = []
    geval_test_cases = []

    node_child_nodes = []
    node_child_metrics = []
    node_child_test_cases = []

    for node, metric, test_case in zip(valid_nodes, valid_metrics, valid_test_cases):
        if node.child is None:
            # Deterministic score case
            deterministic_nodes.append(node)
            deterministic_metrics.append(metric)
            deterministic_test_cases.append(test_case)
        elif isinstance(node.child, GEval):
            geval_nodes.append(node)
            geval_metrics.append(metric)
            geval_test_cases.append(test_case)
        else:
            # Node child case
            node_child_nodes.append(node)
            node_child_metrics.append(metric)
            node_child_test_cases.append(test_case)

    # Handle deterministic scores
    if deterministic_nodes:
        # Set scores immediately
        for node, metric in zip(deterministic_nodes, deterministic_metrics):
            metric.score = node.score / 10  # Convert to 0-1 range

        # Generate reasons if needed
        if any(metric.include_reason for metric in deterministic_metrics):
            reason_prompts = []
            for node, metric in zip(deterministic_nodes, deterministic_metrics):
                if metric.include_reason:
                    prompt = VerdictNodeTemplate.generate_reason(
                        verbose_steps=metric._verbose_steps,
                        score=metric.score,
                        name=metric.__name__,
                    )
                    reason_prompts.append((node, metric, prompt))

            if reason_prompts:
                prompts = [item[2] for item in reason_prompts]
                model.request_dict.metric_phase = f"verdict_reasons_depth_{depth}"
                responses = model.batch_generate(
                    prompts=prompts,
                    schema=MetricScoreReason
                )
                for (node, metric, _), response in zip(reason_prompts, responses):
                    if isinstance(response, MetricScoreReason):
                        metric.reason = response.reason
                    else:
                        metric.reason = "Error: Invalid response"

        # Add verbose logs
        for node, metric in zip(deterministic_nodes, deterministic_metrics):
            metric._verbose_steps.append(
                construct_node_verbose_log(node, depth)
            )

    # Handle GEval children
    if geval_nodes:
        geval_dataset = deepeval_llm_test_cases_to_dataset(geval_test_cases, "temp_geval_dag")
        geval_dataset.metadata = {"dataset_version_id": model.request_dict.dataset_version_id}

        # Assume same GEval child for simplicity; adjust if needed
        child = geval_nodes[0].child
        if LLMTestCaseParams.EXPECTED_OUTPUT in child.evaluation_params:
            with_reference = True
        else:
            with_reference = False

        geval_metrics_results = custom_batch_generate(
            name=child.name,
            evaluator_info=model.request_dict,
            eval_dataset=geval_dataset,
            with_reference=with_reference,
            evaluation_steps=child.evaluation_steps,
            rubrics=child.rubric
        )

        # Extract individual metrics from MetricsResults
        individual_geval_results = geval_metrics_results.metrics

        for node, metric, geval_result_id in zip(geval_nodes, geval_metrics, individual_geval_results):
            metric.score = individual_geval_results[geval_result_id].score
            if metric.include_reason:
                metric.reason = individual_geval_results[geval_result_id].reason
            metric._verbose_steps.append(
                construct_node_verbose_log(node, depth, individual_geval_results[geval_result_id])
            )

    # Handle node children
    if node_child_nodes:
        child_nodes = [node.child for node in node_child_nodes]
        child_metrics = node_child_metrics  # These should be the same metrics
        child_test_cases = node_child_test_cases
        from .batch_execute import batch_execute_node
        batch_execute_node(
            model,
            child_nodes,
            child_metrics,
            child_test_cases,
            depth + 1
        )
