from typing import Any, Dict, Union, Optional, List
from deepeval.metrics import BaseMetric, BaseConversationalMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase


class MetricsResults:
    def __init__(self,
                 metrics: Dict[str, Union[BaseMetric, BaseConversationalMetric]],
                 dataset_info: Any,
                 run_id: str,
                 name: Optional[str] = None):
        # Handles edge cases where the objects do not have a 'name' attribute
        self.name = name or getattr(list(metrics.values())[0], 'name', None)
        self.dataset_info = dataset_info
        self.run_id = run_id
        self.metrics = metrics
        self.__score = None

    @property
    def score(self) -> float:
        if self.__score is None:
            total_score = sum(metric.score for metric in self.metrics.values())
            self.__score = total_score / len(self.metrics) if self.metrics else 0.0
        return self.__score

    def to_dict(self) -> Dict[str, dict]:
        output = {
            "name": self.name,
            "dataset": self.dataset_info,
            "run_id": self.run_id,
            "results": {
                idx: {"score": metric.score, "reason": metric.reason, "success": metric.success}
                for idx, metric in self.metrics.items()
            }
        }
        return output

    def compute_verbose_output(self, test_case_dict: Dict[str, Union[ConversationalTestCase, LLMTestCase]]) -> Dict[str, Dict]:
        output = {}

        for idx, test_case in test_case_dict.items():
            output[idx] = {
                "input": test_case.input,
                "expected_output": test_case.expected_output,
                "actual_output": test_case.actual_output,
                "score": self.metrics[idx].score,
                "reason": self.metrics[idx].reason,
                "success": self.metrics[idx].success
            }
        return output

    def __str__(self) -> str:
        return f"MetricsResults(name={self.name}, dataset={self.dataset_info})"

    def __len__(self) -> int:
        return len(self.metrics)
