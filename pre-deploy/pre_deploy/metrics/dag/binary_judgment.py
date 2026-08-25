from typing import Dict, List, Optional, Sequence

from deepeval.metrics.dag.nodes import (
    BinaryJudgementNode,
    TaskNode,
    decrement_indegree,
    construct_node_verbose_log
)
from deepeval.metrics import BaseMetric
from deepeval.metrics.dag.templates import BinaryJudgementTemplate
from deepeval.metrics.dag.schema import BinaryJudgementVerdict
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.g_eval.utils import G_EVAL_PARAMS

from ...query_processor import DeepEvalClient


def binary_judgment_node_batch_generate(model: DeepEvalClient,
                                        nodes: List[BinaryJudgementNode],
                                        metrics: Sequence[BaseMetric],
                                        test_cases: List[LLMTestCase],
                                        depth: int):
    # Collect prompts only for nodes that can execute (indegree == 0)
    valid_items = []
    valid_test_cases = []
    prompts = []
    for node, metric, test_case in zip(nodes, metrics, test_cases):
        node._depth = max(0, node._depth, depth)
        decrement_indegree(node)
        if node._indegree > 0:
            continue

        text = ""
        if node._parents is not None:
            for parent in node._parents:
                if isinstance(parent, TaskNode):
                    text += f"{parent.output_label}:\n{parent._output}\n"  # type: ignore

        if node.evaluation_params is not None:
            for param in node.evaluation_params:
                value = getattr(test_case, param.value)
                if isinstance(value, ToolCall):
                    value = repr(value)
                text += f"{G_EVAL_PARAMS[param]}:\n{value}\n"

        prompt = BinaryJudgementTemplate.generate_binary_verdict(
            criteria=node.criteria, text=text
        )
        prompts.append(prompt)
        valid_items.append((node, metric))  # Track valid node-metric pairs
        valid_test_cases.append(test_case)

    # Batch generate if there are prompts
    if prompts:
        model.request_dict.metric_phase = f"binary_judgments_depth_{depth}"
        responses = model.batch_generate(
            prompts=prompts,
            schema=BinaryJudgementVerdict
        )
        for (node, metric), response in zip(valid_items, responses):
            if isinstance(response, BinaryJudgementVerdict):
                node._verdict = response
            else:
                node._verdict = BinaryJudgementVerdict(
                    verdict=False, reason="Unable to parse response."
                )

            # Append verbose log per node-metric pair
            metric._verbose_steps.append(
                construct_node_verbose_log(node, node._depth)
            )

    # Recursion to children: Collect all children from processed nodes and batch their execution
    child_nodes = []
    child_metrics = []
    child_test_cases = []
    for (node, metric), test_case in zip(valid_items, valid_test_cases):
        for child in node.children:
            child_nodes.append(child)
            child_metrics.append(metric)
            child_test_cases.append(test_case)
    if child_nodes:
        from .batch_execute import batch_execute_node
        batch_execute_node(
            model,
            child_nodes,
            child_metrics,
            child_test_cases,
            depth + 1
        )
