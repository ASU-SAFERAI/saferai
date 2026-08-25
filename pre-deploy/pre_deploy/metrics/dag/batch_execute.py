from typing import Dict, List, Optional, Sequence

from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.metrics.dag.nodes import (
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode
)

from ...query_processor import DeepEvalClient
from .task_node import task_node_batch_generate
from .binary_judgment import binary_judgment_node_batch_generate
from .nonbinary_judgment_node import nonbinary_judgment_node_batch_generate
from .verdict_node import verdict_node_batch_generate


def batch_execute_node(model: DeepEvalClient, nodes: List,
                       metrics: Sequence[BaseMetric],
                       test_cases: List[LLMTestCase], depth: int):
    if not nodes:
        return
    # Group nodes by type
    grouped = {
        TaskNode: ([], [], []),
        BinaryJudgementNode: ([], [], []),
        NonBinaryJudgementNode: ([], [], []),
        VerdictNode: ([], [], [])
    }
    for node, metric, test_case in zip(nodes, metrics, test_cases):
        node_type = type(node)
        if node_type in grouped:
            grouped[node_type][0].append(node)
            grouped[node_type][1].append(metric)
            grouped[node_type][2].append(test_case)
        else:
            raise TypeError(f"Unsupported node type: {node_type}")

    # Call batch functions for each group
    if grouped[TaskNode][0]:
        task_node_batch_generate(
            model,
            grouped[TaskNode][0],
            grouped[TaskNode][1],
            grouped[TaskNode][2],
            depth
        )
    if grouped[BinaryJudgementNode][0]:
        binary_judgment_node_batch_generate(
            model,
            grouped[BinaryJudgementNode][0],
            grouped[BinaryJudgementNode][1],
            grouped[BinaryJudgementNode][2],
            depth
        )
    if grouped[NonBinaryJudgementNode][0]:
        nonbinary_judgment_node_batch_generate(
            model,
            grouped[NonBinaryJudgementNode][0],
            grouped[NonBinaryJudgementNode][1],
            grouped[NonBinaryJudgementNode][2],
            depth
        )
    if grouped[VerdictNode][0]:
        verdict_node_batch_generate(
            model,
            grouped[VerdictNode][0],
            grouped[VerdictNode][1],
            grouped[VerdictNode][2],
            depth
        )
