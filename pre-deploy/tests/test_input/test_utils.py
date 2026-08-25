"""
Unit tests for input utils helpers.
"""

import unittest
import sys
from pathlib import Path

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.input import build_eval_dataset, build_golden_test_set


class TestInputUtils(unittest.TestCase):
    def test_build_eval_dataset_requires_dataset_version_id(self):
        responses = {"0": "A"}
        dataset = {"0": {"question": "Q"}}

        def question_func(record):
            return record["question"]

        with self.assertRaises(ValueError) as context:
            build_eval_dataset(responses, dataset, question_func, metadata={})
        self.assertIn("metadata must include 'dataset_version_id'", str(context.exception))

    def test_build_eval_dataset_metadata_must_be_dict(self):
        responses = {"0": "A"}
        dataset = {"0": {"question": "Q"}}

        def question_func(record):
            return record["question"]

        with self.assertRaises(ValueError) as context:
            build_eval_dataset(responses, dataset, question_func, metadata="not_a_dict")
        self.assertIn("metadata must be a dictionary", str(context.exception))

    def test_build_eval_dataset_timestamp_id_and_metadata(self):
        responses = {"0": "A"}
        dataset = {"0": {"question": "Q"}}
        metadata = {"dataset_version_id": None, "source": "unit-test"}

        def question_func(record):
            return record["question"]

        eval_dataset = build_eval_dataset(responses, dataset, question_func, metadata=metadata)

        self.assertEqual(len(eval_dataset.id), 14)
        self.assertTrue(eval_dataset.id.isdigit())
        self.assertEqual(eval_dataset.metadata, metadata)
        self.assertEqual(len(eval_dataset.conversations), 1)
        self.assertEqual(eval_dataset.conversations[0].id, "0")

    def test_build_golden_test_set_timestamp_id(self):
        dataset = {
            "0": {"input": "Q", "expected_output": "A"}
        }

        def input_func(record):
            return record["input"]

        def expected_output_func(record):
            return record["expected_output"]

        golden_test_set = build_golden_test_set(dataset, input_func, expected_output_func)

        self.assertEqual(len(golden_test_set.id), 14)
        self.assertTrue(golden_test_set.id.isdigit())
        self.assertEqual(len(golden_test_set.golden_pairs), 1)
        self.assertEqual(golden_test_set.golden_pairs[0].id, "0")
        self.assertEqual(golden_test_set.golden_pairs[0].input, "Q")
        self.assertEqual(golden_test_set.golden_pairs[0].expected_output, "A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
