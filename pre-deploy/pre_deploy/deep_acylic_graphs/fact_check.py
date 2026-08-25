from deepeval.metrics.dag.nodes import (
    BinaryJudgementNode,
    VerdictNode,
    TaskNode
)
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.test_case import LLMTestCaseParams

__fact_detection_node = BinaryJudgementNode(
        criteria="Are all assertions in `facts` accurate?",
        children=[
            VerdictNode(verdict=True, score=10),
            VerdictNode(verdict=False, score=0)
        ]
    )

__extract_fact_node = TaskNode(
    instructions="Extract all facts and assertions stated in `actual_output` in response to `input` in the "
                    "format of a numbered list.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="facts",
    children=[__fact_detection_node]
)

FACT_CHECKER_DAG = DeepAcyclicGraph(root_nodes=[__extract_fact_node])
