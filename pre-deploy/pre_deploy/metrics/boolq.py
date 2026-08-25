from datasets import load_dataset
from datasets import Dataset

import logging
from typing import Optional, Dict, Any, Union

from .. import EvalDataset, DeepEvalClient, GoldenTestSet, AWSEnvironment
from ..query_processor import RequestDict
from ..input.utils import build_eval_dataset, build_golden_test_set
from .utils import _build_metric_request_dict


logger = logging.getLogger(__name__)


class BoolQ:

    def __init__(self,
                 name: str = "reading_comprehension",
                 data: Optional[EvalDataset] = None,
                 evaluator_info: Optional[RequestDict] = None,
                 perturbed: bool = False,
                 environment: Optional[AWSEnvironment] = None):
        super().__init__()
        self.data = data
        self.name = name
        self.dataset_version_id = "0.0.0"
        if environment is None:
            environment = AWSEnvironment(target_account_id=None, role_name=None)
        self.evaluator = DeepEvalClient(
            _build_metric_request_dict(
                evaluator_info=evaluator_info,
                metric_name=name,
                num_eval=0 if self.data is None else len(self.data.conversations),
                dataset_version_id=self.dataset_version_id,
            ),
            environment=environment
        ) if evaluator_info else None
        self.perturbed = perturbed
        self.run_id = None
        self.__results = None
        self.__eval_responses = None

    @property
    def eval_responses(self) -> dict:
        if self.__eval_responses is None:
            eval_queries = self._compute_eval_queries()
            if self.perturbed:
                metric_phase = "perturbed_eval"
            else:
                metric_phase = "eval"
            self.evaluator.request_dict.metric_phase = metric_phase
            self.evaluator.request_dict.dataset_version_id = self.dataset_version_id
            self.__eval_responses = self.evaluator.batch_generate(eval_queries)
            self.run_id = self.evaluator.request_dict.run_id
        return self.__eval_responses

    @property
    def score(self):
        return self.results['exact_match_score']

    @property
    def refusal_rate(self):
        return self.results['refusal_rate']

    @property
    def results(self) -> Dict[str, Any]:
        if self.__results is None:
            self.__results = self._compute_results()
        return self.__results

    def load_boolq_data(self) -> Dataset:
        boolq_data = load_dataset("google/boolq", "default")
        return boolq_data['train']

    def set_eval_dataset(self, responses: dict, boolq_dataset: Dataset = None) -> EvalDataset:
        boolq_dataset = boolq_dataset or self.load_boolq_data()

        eval_dataset = build_eval_dataset(responses, boolq_dataset, self.compute_question,
                                          metadata={'dataset_version_id': self.dataset_version_id})
        self.evaluator.request_dict.num_eval = len(eval_dataset)
        setattr(self, "data", eval_dataset)

    @staticmethod
    def compute_question(data_record):
        return f"{data_record['passage']}\nQuestion: {data_record['question']}?"

    @staticmethod
    def compute_expected_answer(data_record):
        return str(data_record['answer'])

    def _compute_eval_queries(self):
        queries = {}

        for conversation in self.data.conversations:
            question = conversation.messages[0].contents[0].content
            actual_response = conversation.messages[1].contents[0].content
            queries[conversation.id] = f"""
            Based on the following <Question> and <Answer>, classify the answer into one of the categories listed in <Categories>.
            Empty response should be classified as idk. No explanation is needed. Respond with True, False or idk only.
            Limit your response to 5 characters.
            <Question> {question}</Question>
            <Categories> idk; True; False</Categories>
            <Answer> {actual_response}</Answer>
            <Result>
            """

        return queries

    def _compute_results(self) -> Dict[str, Any]:
        total_count = len(self.data.conversations)
        total_correct = 0
        idk_count = 0

        for conversation in self.data.conversations:
            expected_answer = str(conversation.metadata['answer'])
            actual_answer = str(self.eval_responses[conversation.id])
            total_correct += expected_answer == actual_answer
            idk_count += actual_answer == "idk"

        exact_match_score = total_correct / total_count if total_count > 0 else 0
        refusal_rate = idk_count / total_count if total_count > 0 else 0

        return {
            "total_count": len(self.data.conversations),
            "correct_count": total_correct,
            "idk_count": idk_count,
            "exact_match_score": exact_match_score,
            "refusal_rate": refusal_rate,
            "dataset_version_id": self.dataset_version_id,
            "run_id": self.run_id
        }

    def start_batch_generate(self, force_rerun: bool = False) -> Dict[str, Any]:
        eval_queries = self._compute_eval_queries()
        self.evaluator.request_dict.metric_phase = "perturbed_eval" if self.perturbed else "eval"
        self.evaluator.request_dict.dataset_version_id = self.dataset_version_id
        metadata = self.evaluator.enqueue_batch(eval_queries, force_rerun=force_rerun)
        self.run_id = self.evaluator.request_dict.run_id
        return metadata

    def advance_batch_generate(self, phase: str = "eval", force_rerun: bool = False) -> Dict[str, Any]:
        expected_phase = "perturbed_eval" if self.perturbed else "eval"
        if phase != expected_phase:
            raise ValueError(f"phase must be: {expected_phase}")
        return self.start_batch_generate(force_rerun=force_rerun)

    def finalize_batch_generate(self) -> Union[Dict[str, Any]]:
        eval_queries = self._compute_eval_queries()
        self.evaluator.request_dict.metric_phase = "perturbed_eval" if self.perturbed else "eval"
        self.evaluator.request_dict.dataset_version_id = self.dataset_version_id
        responses = self.evaluator.finalize_batch(eval_queries, allow_partial=False)

        if "is_complete" in responses and responses["is_complete"] is False:
            return responses

        self.__eval_responses = responses
        self.run_id = self.evaluator.request_dict.run_id
        return self.results
