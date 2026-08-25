from datasets import Dataset
from typing import Dict, Union, Callable, Optional
from datetime import datetime
from .dataset import EvalDataset
from .golden_test_set import GoldenPair, GoldenTestSet


def build_eval_dataset(responses: Dict[str, str],
                       dataset: Union[Dataset, GoldenTestSet],
                       question_func: Callable,
                       metadata: Dict[str, Optional[str]]) -> EvalDataset:
    # Placing a UUID here is confusing for ET AI Acceleration engineering
    # in DynamoDB, so we will use a timestamp instead.
    eval_dataset_dict = {"id": datetime.now().strftime('%Y%m%d%H%M%S'), "conversations": []}
    if isinstance(metadata, dict):
        # Making dataset_version_id a mandatory yet nullable field in metadata.
        if 'dataset_version_id' not in metadata:
            raise ValueError("metadata must include 'dataset_version_id'")
        eval_dataset_dict["metadata"] = metadata
    else:
        raise ValueError("metadata must be a dictionary")

    # This lookup will not work with augmented GoldenTestSets, since the
    # GoldenPairs are an ordered list and the pairs' IDs are not necessarily in order.
    def _get_item(idx: Union[str, int], dataset: Union[Dataset, GoldenTestSet]):
        try:
            return dataset[idx]
        except (KeyError, IndexError, TypeError):
            try:
                return dataset[str(idx)]
            except (KeyError, IndexError, TypeError):
                try:
                    return dataset[int(idx)]
                except (KeyError, IndexError, TypeError):
                    return None

    def _append_qa_pair(idx, question, response, metadata):
        dict_item = {
            "id": str(idx),
            "messages": [
                {
                    "sequence": 0,
                    "role": "user",
                    "contents": [{"type": "text", "content": question}],
                },
                {
                    "sequence": 1,
                    "role": "assistant",
                    "contents": [{"type": "text", "content": response}],
                }
            ],
            "metadata": metadata
        }
        eval_dataset_dict["conversations"].append(dict_item)

    if isinstance(dataset, GoldenTestSet):
        # If the dataset is an augmented GoldenTestSet, the responses keys are the IDs
        # of the golden pairs, not the indices of the dataset questions.
        for pair in dataset.golden_pairs:
            question = question_func(pair)
            model_response = responses.get(pair.id, None)

            metadata = pair.to_dict()['metadata']
            metadata['expected_output'] = pair.expected_output
            _append_qa_pair(pair.id, question, model_response, metadata)
    else:
        # If the dataset is a canned Hugging Face Dataset, we will assume that
        # the keys of the responses are the same as the keys of the dataset.
        for idx, response in responses.items():
            data_record = _get_item(idx, dataset)
            question = question_func(data_record)
            model_response = response
            metadata = data_record
            _append_qa_pair(idx, question, model_response, metadata)

    return EvalDataset.from_dict(eval_dataset_dict)


def build_golden_test_set(dataset: Union[Dataset, Dict],
                          input_func: Callable,
                          expected_output_func: Callable) -> GoldenTestSet:
    golden_pairs = []
    for idx in dataset:
        data_record = dataset[idx]

        golden_pairs.append(GoldenPair.from_dict({
            "id": idx,
            "input": input_func(data_record),
            "expected_output": expected_output_func(data_record),
            "metadata": data_record
        }))

    golden_test_set = GoldenTestSet(id=datetime.now().strftime('%Y%m%d%H%M%S'),
                                    golden_pairs=golden_pairs)
    return golden_test_set
