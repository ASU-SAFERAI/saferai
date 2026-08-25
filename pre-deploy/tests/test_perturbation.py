import unittest
import sys
from pathlib import Path
import logging
import uuid
from unittest.mock import patch, Mock

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from pre_deploy.input import GoldenTestSet
from pre_deploy.query_processor import RequestDict
from pre_deploy.perturbation import (
    start_generate_perturbed_data_llm,
    advance_generate_perturbed_data_llm,
    finalize_generate_perturbed_data_llm,
)

logging.getLogger('pre_deploy').setLevel(logging.INFO)


class TestPerturbation(unittest.TestCase):
    def test_gender_perturbation(self):
        from pre_deploy.perturbation import gender_perturbation
        golden_test_set = GoldenTestSet.from_dict({
            "id": "test1",
            "data": [
                {
                    "id": "0",
                    "input": "Professor Smith is a male. Right?",
                    "expected_output": "Yes. Professor Smith is a male."
                }
            ]
        })
        perturbed_data = gender_perturbation(golden_test_set=golden_test_set)
        self.assertEqual(perturbed_data["0"].input, "Professor Smith is a female. Right?")
        self.assertEqual(perturbed_data["0"].expected_output, "Yes. Professor Smith is a female.")

    def test_ave_perturbation(self):
        from pre_deploy.perturbation import ave_perturbation
        golden_test_set = GoldenTestSet.from_dict({
            "id": "test2",
            "data": [
                {
                    "id": "0",
                    "input": "Baby, come back to Mom.",
                    "expected_output": ""
                }
            ]
        })
        perturbed_data = ave_perturbation(golden_test_set=golden_test_set)
        self.assertEqual(perturbed_data["0"].input, "Baby, cum back to Momma.")
        self.assertEqual(perturbed_data["0"].expected_output, "")

    def test_typos_perturbation(self):
        from pre_deploy.perturbation import typos_perturbation

        golden_test_set = GoldenTestSet.from_dict({
            "id": "test3",
            "data": [
                {
                    "id": "0",
                    "input": "This is a test sentence.",
                    "expected_output": ""
                }
            ]
        })
        perturbed_data = typos_perturbation(golden_test_set=golden_test_set, prob=0.1)

        def _char_overlap(a: str, b: str) -> float:
            if not a and not b:
                return 1.0
            max_len = max(len(a), len(b))
            matches = sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i])
            return matches / max_len

        self.assertAlmostEqual(_char_overlap(perturbed_data["0"].input, golden_test_set["0"].input), 0.9, places=1)

    def test_synonym_perturbation(self):
        from pre_deploy.perturbation import synonym_perturbation
        golden_test_set = GoldenTestSet.from_dict({
            "id": "test4",
            "data": [
                {
                    "id": "0",
                    "input": "I believe I am right on this one.",
                    "expected_output": ""
                }
            ]
        })
        perturbed_data = synonym_perturbation(golden_test_set=golden_test_set, prob=0.5)

        def _word_overlap(a: str, b: str) -> float:
            a_words = a.split()
            b_words = b.split()
            if not a_words and not b_words:
                return 1.0
            matches = sum(1 for i, w in enumerate(a_words) if i < len(b_words) and b_words[i] == w)
            return matches / max(len(a_words), len(b_words))

        overlap_pct = _word_overlap(perturbed_data["0"].input, golden_test_set["0"].input)

        self.assertTrue(overlap_pct > 0.4 and overlap_pct < 1.0)

    def test_llm_perturbation_single(self):
        from pre_deploy.perturbation import generate_perturbed_data_llm
        query_dict = {
            "0": "The quick brown fox jumps over the lazy dog."
        }
        perturbed_query = generate_perturbed_data_llm(request_dict=RequestDict(
                                                        run_id=str(uuid.uuid4()),
                                                        username="test_user",
                                                        metric_name="perturbation",
                                                        metric_phase="perturbation",
                                                        model_name="gpt4o_mini",
                                                        model_provider="openai",
                                                        num_eval=len(query_dict)
                                                      ),
                                                      query_dict=query_dict,
                                                      perturbation_types=["50% synonym replacement", "50% typo replacement"])
        self.assertTrue(type(perturbed_query) is dict)
        self.assertIn("0", perturbed_query)
        self.assertTrue(type(perturbed_query['0']) is str)

    @patch('pre_deploy.perturbation.llm_perturbation.QueryProcessorClient.enqueue_phase', autospec=True)
    def test_start_enqueues_perturbation_phase(self, mock_enqueue_phase):
        query_dict = {"0": "The quick brown fox jumps over the lazy dog."}
        request_dict = RequestDict(
            run_id=str(uuid.uuid4()),
            username="test_user",
            metric_name="perturbation",
            metric_phase="perturbation",
            model_name="gpt4o_mini",
            model_provider="openai",
            num_eval=len(query_dict),
        )

        mock_enqueue_phase.return_value = {
            "run_id": request_dict.run_id,
            "metric_phase": "perturbation",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = start_generate_perturbed_data_llm(
            request_dict=request_dict,
            query_dict=query_dict,
            perturbation_types=["50% synonym replacement", "50% typo replacement"],
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "perturbation")
        self.assertTrue(mock_enqueue_phase.called)
        client_self = mock_enqueue_phase.call_args[0][0]
        self.assertEqual(client_self.request_dict.metric_phase, "perturbation")

    @patch('pre_deploy.perturbation.llm_perturbation.QueryProcessorClient.enqueue_phase', autospec=True)
    def test_advance_reenqueues_matching_phase(self, mock_enqueue_phase):
        query_dict = {"0": "The quick brown fox jumps over the lazy dog."}
        request_dict = RequestDict(
            run_id=str(uuid.uuid4()),
            username="test_user",
            metric_name="perturbation",
            metric_phase="perturbation",
            model_name="gpt4o_mini",
            model_provider="openai",
            num_eval=len(query_dict),
        )

        mock_enqueue_phase.return_value = {
            "run_id": request_dict.run_id,
            "metric_phase": "perturbation",
            "total_items": 1,
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = advance_generate_perturbed_data_llm(
            request_dict=request_dict,
            query_dict=query_dict,
            perturbation_types=["50% synonym replacement", "50% typo replacement"],
            phase="perturbation",
            environment=Mock(),
        )

        self.assertEqual(result["metric_phase"], "perturbation")
        self.assertTrue(mock_enqueue_phase.called)

    def test_advance_raises_for_mismatched_phase(self):
        query_dict = {"0": "The quick brown fox jumps over the lazy dog."}
        request_dict = RequestDict(
            run_id=str(uuid.uuid4()),
            username="test_user",
            metric_name="perturbation",
            metric_phase="perturbation",
            model_name="gpt4o_mini",
            model_provider="openai",
            num_eval=len(query_dict),
        )

        with self.assertRaises(ValueError):
            advance_generate_perturbed_data_llm(
                request_dict=request_dict,
                query_dict=query_dict,
                perturbation_types=["50% synonym replacement", "50% typo replacement"],
                phase="wrong_phase",
                environment=Mock(),
            )

    @patch('pre_deploy.perturbation.llm_perturbation.QueryProcessorClient.finalize_phase', autospec=True)
    def test_finalize_returns_filtered_responses_when_complete(self, mock_finalize_phase):
        query_dict = {
            "0": "The quick brown fox jumps over the lazy dog.",
            "1": "Keep this sentence unchanged.",
        }
        request_dict = RequestDict(
            run_id=str(uuid.uuid4()),
            username="test_user",
            metric_name="perturbation",
            metric_phase="perturbation",
            model_name="gpt4o_mini",
            model_provider="openai",
            num_eval=len(query_dict),
        )

        mock_finalize_phase.return_value = {
            "responses": {
                "0": "A fast brown fox jumps over the lazy dog.",
                "1": "Keep this sentence unchanged.",
            },
            "completed_items": 2,
            "pending_items": 0,
            "is_complete": True,
        }

        result = finalize_generate_perturbed_data_llm(
            request_dict=request_dict,
            query_dict=query_dict,
            perturbation_types=["50% synonym replacement", "50% typo replacement"],
            filter_no_change=True,
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        self.assertIn("0", result)
        self.assertNotIn("1", result)

    @patch('pre_deploy.perturbation.llm_perturbation.QueryProcessorClient.finalize_phase', autospec=True)
    def test_finalize_returns_incomplete_status_when_phase_incomplete(self, mock_finalize_phase):
        query_dict = {"0": "The quick brown fox jumps over the lazy dog."}
        request_dict = RequestDict(
            run_id=str(uuid.uuid4()),
            username="test_user",
            metric_name="perturbation",
            metric_phase="perturbation",
            model_name="gpt4o_mini",
            model_provider="openai",
            num_eval=len(query_dict),
        )

        mock_finalize_phase.return_value = {
            "responses": {"0": None},
            "completed_items": 0,
            "pending_items": 1,
            "is_complete": False,
        }

        result = finalize_generate_perturbed_data_llm(
            request_dict=request_dict,
            query_dict=query_dict,
            perturbation_types=["50% synonym replacement", "50% typo replacement"],
            environment=Mock(),
        )

        self.assertIsInstance(result, dict)
        self.assertFalse(result.get("is_complete"))
        self.assertEqual(result.get("pending_items"), 1)
        self.assertEqual(result.get("completed_items"), 0)

if __name__ == '__main__':
    unittest.main()
