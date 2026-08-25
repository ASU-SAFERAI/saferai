from typing import Any, Dict, List, Optional, cast

from deepeval.metrics.utils import (
    check_llm_test_case_params,
)
from deepeval.metrics import DAGMetric
from deepeval.metrics.dag.graph import DeepAcyclicGraph
from deepeval.metrics.dag.utils import (
    extract_required_params,
)

from ...input import EvalDataset
from ...loaders.deepeval_test_cases import dataset_to_deepeval_llm_test_cases
from ...output.results import MetricsResults
from ...query_processor import DeepEvalClient
from .batch_execute import batch_execute_node


def dag_batch_generate(
    model: DeepEvalClient,
    eval_dataset: EvalDataset,
    name: str,
    dag: DeepAcyclicGraph,
    threshold: float = 0.5,
    include_reason: bool = True,
    strict_mode: bool = False,
) -> MetricsResults:
    dataset_version_id = eval_dataset.metadata.get('dataset_version_id')
    if not isinstance(dataset_version_id, str) or len(dataset_version_id) <= 1:
        raise ValueError(
            "eval_dataset.metadata['dataset_version_id'] must be a string with length > 1 "
            "(use 'null' when no dataset version exists)."
        )
    model.request_dict.dataset_version_id = dataset_version_id

    test_cases_dict = dataset_to_deepeval_llm_test_cases(eval_dataset)

    metrics = {
        convo.id: DAGMetric(
            name=name,
            dag=dag,
            model=model,
            threshold=threshold,
            include_reason=include_reason,
            async_mode=False,
            strict_mode=strict_mode,
            verbose_mode=False
        )
        for convo in eval_dataset.conversations
    }

    for idx, metric in metrics.items():
        check_llm_test_case_params(
            test_cases_dict[idx],
            list(extract_required_params(nodes=metric.dag.root_nodes)),
            metric
        )

    # Collect all root nodes, metrics, and test_cases
    all_roots = [root for metric in metrics.values()
                 for root in metric.dag.root_nodes]

    # Call batch_execute_node with all roots
    batch_execute_node(
        model,
        all_roots,
        list(metrics.values()),
        list(test_cases_dict.values()),
        0
    )

    for _, metric in metrics.items():
        metric.success = metric.is_successful()

    # Return MetricsResults object for analytics
    return MetricsResults(
        name=name,
        dataset_info=eval_dataset.metadata,
        run_id=model.request_dict.run_id,
        metrics=cast(Dict[str, Any], metrics)
    )
