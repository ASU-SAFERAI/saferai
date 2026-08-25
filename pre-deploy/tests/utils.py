from typing import Any, Callable, Dict, List, Optional
import uuid

from pre_deploy import EvalDataset


INVALID_DATASET_VERSION_ID_METADATAS: List[Optional[Dict[str, Any]]] = [
    None,
    {},
    {"dataset_version_id": None},
    {"dataset_version_id": 1},
    {"dataset_version_id": ""},
    {"dataset_version_id": "x"},
]


def assert_invalid_dataset_version_id_raises(
    test_case: Any,
    test_set: List[Dict[str, Any]],
    create_dataset: Callable[..., EvalDataset],
    run_metric_with_dataset: Callable[[EvalDataset], Any],
) -> None:
    for metadata in INVALID_DATASET_VERSION_ID_METADATAS:
        with test_case.subTest(metadata=metadata):
            eval_dataset = (
                create_dataset(test_set)
                if metadata is None
                else create_dataset(test_set, metadata=metadata)
            )

            with test_case.assertRaises(ValueError) as context:
                run_metric_with_dataset(eval_dataset)

            test_case.assertIn("dataset_version_id", str(context.exception))
            test_case.assertIn("length > 1", str(context.exception))


def create_eval_dataset_for_testing(test_set: List[Dict], metadata: Optional[Dict] = None):
    """
    This is a util function for building inputs for single-turn metric unit tests

    test_set sample:
    [
        {
            "input": "Hello!",
            "actual_output": "Hi! How can I help?",
            "expected_output": "Hi! How can I assist you today?",
            "context": ["greeting", "casual"]
            ...
        },
        ...
    ]

    """
    eval_dataset_dict = {"id": str(uuid.uuid4()), "conversations": []}
    if metadata:
        eval_dataset_dict["metadata"] = metadata

    idx = 0
    for record in test_set:
        dict_item = {
            "id": str(idx),
            "messages": [
                {
                    "sequence": 0,
                    "role": "user",
                    "contents": [{"type": "text", "content": record["input"]}],
                },
                {
                    "sequence": 1,
                    "role": "assistant",
                    "contents": [{"type": "text", "content": record["actual_output"]}],
                }
            ],
            "metadata": record
        }
        idx += 1

        if "expected_output" in record:
            dict_item["metadata"] = {
                "expected_output": record["expected_output"]
            }
        if "context" in record:
            dict_item["messages"][1]["metadata"] = {
                "context": record["context"]
            }

        eval_dataset_dict["conversations"].append(dict_item)

    return EvalDataset.from_dict(eval_dataset_dict)


def create_eval_conversation_dataset_for_testing(test_set: List[Dict], metadata: Optional[Dict] = None):
    """
    This is a utility function for building conversational test datasets.
    While this is similar to its single turn counterpart, it ends up generating one conversation
    for all turns rather than multiple conversation IDs.
    """
    eval_dataset_dict = {"id": str(uuid.uuid4()), "conversations": []}
    if metadata:
        eval_dataset_dict["metadata"] = metadata

    idx = 0
    conv_dict_item = {
        "id": str(0),
        "messages": [],
        "metadata": {}
    }
    for record in test_set:
        conv_dict_item['messages'].append(
            {
                "sequence": idx,
                "role": "user",
                "contents": [{"type": "text", "content": record["input"]}],
            },
        )
        idx += 1
        conv_dict_item['messages'].append(
            {
                "sequence": idx,
                "role": "assistant",
                "contents": [{"type": "text", "content": record["actual_output"]}],
            }
        )
        idx += 1

    eval_dataset_dict["conversations"] = [conv_dict_item]
    return EvalDataset.from_dict(eval_dataset_dict)
