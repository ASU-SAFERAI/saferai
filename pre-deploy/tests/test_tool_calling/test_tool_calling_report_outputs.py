import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from custom_benchmarks.tool_calling.models import CanonicalCall, ToolCallObservation


class TestToolCallingReportOutputs(unittest.TestCase):
    @staticmethod
    def _frames():
        detail = pd.DataFrame(
            [
                {
                    "report_schema_version": 7,
                    "run_id": "run-1",
                    "model": "candidate",
                    "provider": "provider",
                    "scenario_id": "scenario-1",
                    "question": "find dates",
                    "enabled_tools": {"websearch": True},
                    "expected_calls": [{"tool_name": "websearch", "arguments": {"query": "find"}}],
                    "observed_calls": '[{"tool_name":"websearch"}]',
                    "tool_outputs": '[{"results":["r"]}]',
                    "final_response": "done",
                    "request": {"query": "find"},
                    "stream_summary": {"frame_count": 2},
                    "score": 1.0,
                    "success": True,
                    "evaluation_status": "scored",
                    "meta_construct_type": "discrimination",
                }
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "model": "candidate",
                    "provider": "provider",
                    "questions": 1,
                    "scored_questions": 1,
                    "mean_score": 1.0,
                    "pass_rate": 1.0,
                }
            ]
        )
        return detail, summary

    @staticmethod
    def _observation():
        return ToolCallObservation(
            run_id="run-1",
            scenario_id="scenario-1",
            candidate={"name": "candidate", "provider": "provider"},
            request={"query": "find", "access_token": "secret-token"},
            status="completed",
            completed=True,
            eos_received=True,
            tool_calls=[
                CanonicalCall(
                    ordinal=0,
                    tool_name="websearch",
                    arguments={"query": "find"},
                    tool_output={"results": ["result"]},
                    raw={"authorization": "Bearer secret-token"},
                )
            ],
            final_response="done",
            stream_summary={"frame_count": 2},
            raw_trace={"frames": [{"authorization": "Bearer secret-token"}]},
        )

    def test_manifest_is_written_and_raw_observations_are_opt_in(self):
        detail, summary = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            paths = benchmark.write_outputs(detail, summary, Path(directory))

            self.assertTrue(paths["manifest"].is_file())
            self.assertIsNone(paths["observations_json"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["manifest_schema_version"], 1)
            self.assertEqual(manifest["artifacts"]["detail_csv"], str(paths["detail_csv"]))
            self.assertNotIn("observations_json", manifest["artifacts"])

    def test_opt_in_raw_observations_include_manifest_links_and_redacted_values(self):
        detail, summary = self._frames()
        manifest = benchmark.build_run_manifest(
            run_id="run-1",
            dataset={"path": "dataset.json", "version": "dataset-hash", "scenario_count": 1},
            candidate_models=[{"name": "candidate", "provider": "provider"}],
            evaluator_model={"name": "judge", "provider": "judge-provider"},
            timeout=12.5,
            execution_policy={"concurrency": 1, "retry_count": 0},
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:01:00Z",
            websocket_config={
                "authenticated_ws_url": "wss://example.test/query?access_token=secret-token"
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = benchmark.write_outputs(
                detail,
                summary,
                Path(directory),
                observations=[self._observation()],
                run_manifest=manifest,
                include_raw_observations=True,
            )

            self.assertTrue(paths["observations_json"].is_file())
            payload = json.loads(paths["observations_json"].read_text(encoding="utf-8"))
            self.assertEqual(payload["manifest"]["run_id"], "run-1")
            self.assertEqual(
                payload["manifest"]["artifacts"]["observations_json"],
                str(paths["observations_json"]),
            )
            observation = payload["observations"][0]
            self.assertEqual(observation["request"]["access_token"], "<redacted>")
            self.assertNotIn("secret-token", json.dumps(payload))
            self.assertIn("raw_trace", observation)

        detail, summary = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            paths = benchmark.write_outputs(detail, summary, Path(directory))

            self.assertTrue(paths["detail_csv"].is_file())
            self.assertTrue(paths["summary_csv"].is_file())
            self.assertEqual(paths["detail_csv"].parent, Path(directory))
            self.assertIn("tool_calling_benchmark_detail_", paths["detail_csv"].name)
            self.assertIn("tool_calling_benchmark_summary_", paths["summary_csv"].name)
            self.assertNotEqual(paths["detail_csv"].stem, paths["summary_csv"].stem)

            detail_frame = pd.read_csv(paths["detail_csv"])
            detail_row = detail_frame.iloc[0]
            summary_row = pd.read_csv(paths["summary_csv"]).iloc[0]
            self.assertEqual(summary_row["report_schema_version"], 1)
            # Audit view retains identity, dataset metadata, ground truth,
            # observed calls, and the verdict.
            self.assertEqual(detail_row["run_id"], "run-1")
            self.assertEqual(detail_row["scenario_id"], "scenario-1")
            self.assertEqual(json.loads(detail_row["enabled_tools"]), {"websearch": True})
            self.assertEqual(
                json.loads(detail_row["expected_calls"])[0]["tool_name"],
                "websearch",
            )
            self.assertEqual(detail_row["score"], 1.0)
            self.assertEqual(detail_row["meta_construct_type"], "discrimination")
            # Heavy forensic/transport columns are dropped from the audit CSV.
            self.assertNotIn("request", detail_frame.columns)
            self.assertNotIn("stream_summary", detail_frame.columns)

    def test_artifact_ids_do_not_overwrite_same_second_runs(self):
        detail, summary = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark.pd, "ExcelWriter", side_effect=ImportError):
                first = benchmark.write_outputs(detail, summary, Path(directory))
                second = benchmark.write_outputs(detail, summary, Path(directory))

            self.assertNotEqual(first["detail_csv"], second["detail_csv"])
            self.assertNotEqual(first["summary_csv"], second["summary_csv"])
            self.assertEqual(len(list(Path(directory).glob("*_detail_*.csv"))), 2)
            self.assertEqual(len(list(Path(directory).glob("*_summary_*.csv"))), 2)

    def test_csvs_survive_unavailable_xlsx_support(self):
        detail, summary = self._frames()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(benchmark.pd, "ExcelWriter", side_effect=ImportError("openpyxl missing")):
                paths = benchmark.write_outputs(detail, summary, Path(directory))

            self.assertIsNone(paths["xlsx"])
            self.assertTrue(paths["detail_csv"].is_file())
            self.assertTrue(paths["summary_csv"].is_file())
            self.assertEqual(len(list(Path(directory).glob("*.csv"))), 2)


if __name__ == "__main__":
    unittest.main()
