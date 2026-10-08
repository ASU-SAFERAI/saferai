import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from custom_benchmarks.tool_calling.dataset_validation import DatasetValidationError


class TestToolCallingBenchmarkDryRun(unittest.TestCase):
    @staticmethod
    def _scenario(question="Use the enabled search tool."):
        return {
            "scenario_id": "scenario-1",
            "questions": [question],
            "system_prompt": "Legacy prompt must never be sent.",
            "enabled_tools": {"create_artifact": False, "websearch": True},
            "expected_tools": [],
            "hop_count": 0,
            "golden_truth": [],
        }

    def test_build_dry_run_report_is_ordered_and_excludes_system_prompt(self):
        scenarios = [self._scenario("First question"), self._scenario("Second question")]
        scenarios[1]["scenario_id"] = "scenario-2"
        candidate_models = [
            {"name": "first-model", "provider": "provider-a"},
            {"name": "second-model", "provider": "provider-b"},
        ]

        report = benchmark.build_dry_run_report(
            scenarios,
            candidate_models,
            dataset_path="dataset.json",
        )

        self.assertEqual(report["mode"], "dry-run")
        self.assertEqual(report["request_count"], 4)
        self.assertEqual(
            [(item["model_name"], item["scenario_id"]) for item in report["requests"]],
            [
                ("first-model", "scenario-1"),
                ("first-model", "scenario-2"),
                ("second-model", "scenario-1"),
                ("second-model", "scenario-2"),
            ],
        )
        request = report["requests"][0]["request"]
        self.assertEqual(request["query"], "First question")
        self.assertEqual(request["agent_config"]["tools"], ["websearch"])
        self.assertNotIn("system_prompt", request)
        self.assertNotIn("Legacy prompt must never be sent.", json.dumps(request))

    def test_cli_dry_run_validates_and_never_loads_credentials_or_opens_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(
                json.dumps([self._scenario()]),
                encoding="utf-8",
            )
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path",
                str(dataset_path),
                "--candidate-model",
                "candidate-model",
                "--candidate-provider",
                "candidate-provider",
                "--dry-run",
            ]
            output = StringIO()
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark, "load_credentials") as load_credentials,
                patch.object(benchmark, "AgenticWebSocketClient") as websocket_client,
                redirect_stdout(output),
            ):
                benchmark.main()

        load_credentials.assert_not_called()
        websocket_client.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["candidate_count"], 1)
        self.assertEqual(report["request_count"], 1)
        self.assertEqual(report["requests"][0]["request"]["query"], "Use the enabled search tool.")
        self.assertEqual(report["requests"][0]["request"]["agent_config"]["tools"], ["websearch"])
        self.assertNotIn("system_prompt", report["requests"][0]["request"])

    def test_cli_retains_legacy_flags_and_applies_limit_in_source_order(self):
        scenarios = [self._scenario("First question"), self._scenario("Second question")]
        scenarios[0]["scenario_id"] = "scenario-first"
        scenarios[1]["scenario_id"] = "scenario-second"
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            output_dir = Path(directory) / "results"
            dataset_path.write_text(json.dumps(scenarios), encoding="utf-8")
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path",
                str(dataset_path),
                "--limit",
                "1",
                "--output-dir",
                str(output_dir),
                "--log-level",
                "DEBUG",
                "--candidate-model",
                "candidate-model",
                "--candidate-provider",
                "candidate-provider",
                "--dry-run",
            ]
            output = StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(output):
                benchmark.main()

        report = json.loads(output.getvalue())
        self.assertEqual(report["scenario_count"], 1)
        self.assertEqual(report["request_count"], 1)
        self.assertEqual(report["requests"][0]["scenario_id"], "scenario-first")
        self.assertEqual(report["requests"][0]["request"]["query"], "First question")

    def test_cli_wires_polling_and_raw_output_options(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps([self._scenario()]), encoding="utf-8")
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path", str(dataset_path),
                "--output-dir", str(Path(directory) / "results"),
                "--candidate-model", "candidate-model",
                "--candidate-provider", "candidate-provider",
                "--evaluator-model", "judge-model",
                "--evaluator-provider", "judge-provider",
                "--config-path", str(Path(directory) / "credentials.conf"),
                "--environment", "QA",
                "--timeout", "45",
                "--poll-interval", "2.5",
                "--max-polls", "7",
                "--raw-observations",
                "--raw-frames",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    benchmark,
                    "load_credentials",
                    return_value=SimpleNamespace(
                        authenticated_ws_url="wss://example.invalid/query?access_token=<redacted>"
                    ),
                ),
                patch.object(benchmark, "AgenticWebSocketClient"),
                patch.object(benchmark, "collect_candidate_responses", return_value={}) as collect,
                patch.object(benchmark, "build_trace_eval_dataset", return_value=object()),
                patch.object(benchmark, "score_accuracy", return_value=object()) as score,
                patch.object(
                    benchmark,
                    "build_report",
                    return_value=[
                        {
                            "model": "candidate-model",
                            "provider": "candidate-provider",
                            "run_id": "candidate-run",
                            "scenario_id": "scenario-1",
                            "score": 1.0,
                            "success": True,
                        }
                    ],
                ),
                patch.object(benchmark, "write_outputs") as write_outputs,
            ):
                benchmark.main()

        collect.assert_called_once()
        self.assertTrue(collect.call_args.kwargs["include_raw_frames"])
        score.assert_called_once()
        self.assertEqual(score.call_args.kwargs["poll_interval_seconds"], 2.5)
        self.assertEqual(score.call_args.kwargs["max_polls"], 7)
        write_outputs.assert_called_once()
        self.assertTrue(write_outputs.call_args.kwargs["include_raw_observations"])
        manifest = write_outputs.call_args.kwargs["run_manifest"]
        self.assertEqual(
            manifest["evaluator_polling"],
            {"interval_seconds": 2.5, "max_polls": 7},
        )
        self.assertTrue(manifest["raw_frames"])
        self.assertTrue(manifest["raw_observations"])

    def test_configuration_failure_exits_with_clear_run_level_status(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps([self._scenario()]), encoding="utf-8")
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path", str(dataset_path),
                "--candidate-model", "candidate-model",
                "--candidate-provider", "candidate-provider",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    benchmark,
                    "load_credentials",
                    side_effect=RuntimeError("invalid access token"),
                ),
                patch.object(benchmark, "AgenticWebSocketClient") as websocket_client,
            ):
                with self.assertRaises(SystemExit) as raised:
                    benchmark.main()

        self.assertEqual(raised.exception.code, 2)
        websocket_client.assert_not_called()

    def test_no_reportable_results_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "dataset.json"
            dataset_path.write_text(json.dumps([self._scenario()]), encoding="utf-8")
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path", str(dataset_path),
                "--candidate-model", "candidate-model",
                "--candidate-provider", "candidate-provider",
                "--output-dir", str(Path(directory) / "results"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    benchmark,
                    "load_credentials",
                    return_value=SimpleNamespace(
                        authenticated_ws_url="wss://example.invalid/query?access_token=<redacted>"
                    ),
                ),
                patch.object(benchmark, "AgenticWebSocketClient"),
                patch.object(benchmark, "collect_candidate_responses", return_value={}),
                patch.object(benchmark, "build_trace_eval_dataset", return_value=object()),
                patch.object(
                    benchmark,
                    "score_accuracy",
                    side_effect=RuntimeError("evaluator unavailable"),
                ),
                patch.object(benchmark, "write_outputs") as write_outputs,
            ):
                with self.assertRaises(SystemExit) as raised:
                    benchmark.main()

        self.assertEqual(raised.exception.code, 1)
        write_outputs.assert_not_called()

    def test_cli_dry_run_rejects_invalid_dataset_before_credentials(self):
        invalid_scenario = self._scenario()
        invalid_scenario["enabled_tools"] = {"not-registered": True}
        with tempfile.TemporaryDirectory() as directory:
            dataset_path = Path(directory) / "invalid.json"
            dataset_path.write_text(json.dumps([invalid_scenario]), encoding="utf-8")
            argv = [
                "run_tool_calling_benchmark.py",
                "--dataset-path",
                str(dataset_path),
                "--dry-run",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(benchmark, "load_credentials") as load_credentials,
            ):
                with self.assertRaises(DatasetValidationError):
                    benchmark.main()

        load_credentials.assert_not_called()


if __name__ == "__main__":
    unittest.main()
