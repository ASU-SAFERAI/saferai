import math
import unittest
from unittest.mock import patch

import custom_benchmarks.run_tool_calling_benchmark as benchmark
from custom_benchmarks.tool_calling.agentic import client as client_module
from custom_benchmarks.tool_calling.agentic.client import AgenticWebSocketClient
from custom_benchmarks.tool_calling.agentic.config import (
    WebSocketCredentials,
    load_credentials,
    redact_sensitive_data,
    redact_sensitive_text,
    validate_websocket_timeout,
)


class TestAgenticConfiguration(unittest.TestCase):
    def test_load_credentials_uses_selected_section_without_exposing_secrets(self):
        config = """
[QA]
access_token = qa-secret-token
ws_url = wss://agent.example.invalid/query?access_token=
"""
        with self.subTest("selected config section"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "credentials.conf"
                path.write_text(config, encoding="utf-8")
                credentials = load_credentials(path, environment="QA")

        self.assertIn("qa-secret-token", credentials.authenticated_ws_url)
        self.assertNotIn("qa-secret-token", repr(credentials))
        self.assertEqual(
            redact_sensitive_text(credentials.authenticated_ws_url),
            "wss://agent.example.invalid/query?access_token=<redacted>",
        )

    def test_timeout_must_be_finite_positive_and_bounded(self):
        self.assertEqual(validate_websocket_timeout(30), 30.0)
        for value in (0, -1, math.inf, math.nan, 601):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_websocket_timeout(value)

    def test_redacts_nested_secret_values_and_authenticated_urls(self):
        payload = {
            "query": "normal",
            "headers": {"Authorization": "Bearer header-secret"},
            "nested": {"api_key": "key-secret"},
            "url": "wss://example.invalid/?access_token=url-secret",
        }

        redacted = redact_sensitive_data(payload)

        self.assertEqual(redacted["headers"]["Authorization"], "<redacted>")
        self.assertEqual(redacted["nested"]["api_key"], "<redacted>")
        self.assertNotIn("url-secret", redacted["url"])
        self.assertNotIn("header-secret", str(redacted))
        self.assertNotIn("key-secret", str(redacted))


class TestAgenticClientWiring(unittest.TestCase):
    def test_client_passes_bounded_timeout_and_redacts_request_and_exception(self):
        credentials = WebSocketCredentials(
            access_token="url-secret",
            authenticated_ws_url="wss://example.invalid/query?access_token=url-secret",
        )
        client = AgenticWebSocketClient(credentials, timeout=45)
        request = {
            "query": "run",
            "headers": {"Authorization": "Bearer header-secret"},
        }

        with patch.object(
            client_module.websocket,
            "create_connection",
            side_effect=RuntimeError(
                "failed wss://example.invalid/?access_token=url-secret"
            ),
        ) as create_connection:
            result = client.query(request)

        create_connection.assert_called_once_with(
            credentials.authenticated_ws_url,
            timeout=45.0,
            sslopt=client._sslopt,
        )
        self.assertIn("<redacted>", result.trace.transport_error)
        self.assertNotIn("url-secret", result.trace.transport_error)
        serialized = result.to_dict()
        self.assertEqual(serialized["request"]["headers"]["Authorization"], "<redacted>")
        self.assertNotIn("header-secret", str(serialized))
        self.assertNotIn("url-secret", str(serialized))


class TestBenchmarkModelSelection(unittest.TestCase):
    def test_candidate_override_and_provider_filter_preserve_identity(self):
        self.assertEqual(
            benchmark._select_candidate_models("custom-model", "custom-provider"),
            [{"name": "custom-model", "provider": "custom-provider"}],
        )
        self.assertEqual(
            [model["provider"] for model in benchmark._select_candidate_models(
                model_provider="aws"
            )],
            ["aws", "aws"],
        )

    def test_evaluator_override_and_self_judge_fallback_compare_provider_too(self):
        evaluator = benchmark._resolve_evaluator_model("judge", "judge-provider")
        self.assertEqual(evaluator, {"name": "judge", "provider": "judge-provider"})
        self.assertEqual(
            benchmark._select_evaluator(evaluator, evaluator),
            benchmark.FALLBACK_EVALUATOR_MODEL,
        )
        self.assertEqual(
            benchmark._select_evaluator(benchmark.EVALUATOR_MODEL),
            benchmark.FALLBACK_EVALUATOR_MODEL,
        )
        self.assertEqual(
            benchmark._select_evaluator(
                {"name": "gemma4_31b_it", "provider": "other-provider"}
            ),
            benchmark.EVALUATOR_MODEL,
        )

        self.assertEqual(benchmark.validate_poll_interval_seconds(2), 2.0)
        self.assertEqual(benchmark.validate_max_polls(7), 7)
        for value in (0, -1, float("inf"), float("nan"), 601):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    benchmark.validate_poll_interval_seconds(value)
        for value in (0, -1, benchmark.MAX_ALLOWED_POLLS + 1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    benchmark.validate_max_polls(value)

    def test_cli_parser_preserves_defaults_and_accepts_agentic_controls(self):
        defaults = benchmark.build_argument_parser().parse_args([])
        self.assertIsNone(defaults.config_path)
        self.assertIsNone(defaults.environment)
        self.assertEqual(defaults.timeout, benchmark.DEFAULT_WEBSOCKET_TIMEOUT_SECONDS)
        self.assertEqual(defaults.poll_interval_seconds, benchmark.POLL_INTERVAL_SECONDS)
        self.assertEqual(defaults.max_polls, benchmark.MAX_POLLS)
        self.assertFalse(defaults.raw_observations)
        self.assertFalse(defaults.raw_frames)

        args = benchmark.build_argument_parser().parse_args([
            "--agent-config-path", "credentials.conf",
            "--agent-environment", "QA",
            "--websocket-timeout", "45",
            "--evaluator-poll-interval", "2.5",
            "--evaluator-max-polls", "7",
            "--candidate-model", "candidate",
            "--candidate-provider", "provider",
            "--evaluator-model", "judge",
            "--evaluator-provider", "judge-provider",
            "--raw-output",
            "--include-raw-frames",
        ])
        self.assertEqual(args.config_path.name, "credentials.conf")
        self.assertEqual(args.environment, "QA")
        self.assertEqual(args.timeout, 45.0)
        self.assertEqual(args.poll_interval_seconds, 2.5)
        self.assertEqual(args.max_polls, 7)
        self.assertEqual(args.candidate_model, "candidate")
        self.assertEqual(args.candidate_provider, "provider")
        self.assertEqual(args.evaluator_model, "judge")
        self.assertEqual(args.evaluator_provider, "judge-provider")
        self.assertTrue(args.raw_observations)
        self.assertTrue(args.raw_frames)

    def test_cli_parser_rejects_invalid_polling_bounds(self):
        for option, value in (
            ("--timeout", "0"),
            ("--timeout", "601"),
            ("--poll-interval", "0"),
            ("--max-polls", "0"),
        ):
            with self.subTest(option=option):
                with self.assertRaises(SystemExit):
                    benchmark.build_argument_parser().parse_args([option, value])



if __name__ == "__main__":
    unittest.main()
