"""Benchmark configuration: candidate/evaluator models, bounds, and CLI parser.

Owns the model catalog, evaluator selection rules, polling/execution bounds and
their validators, and the argument parser. These are the values a maintainer
tunes most often, kept separate from execution and reporting logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from custom_benchmarks.tool_calling.agentic.config import (
    DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
    validate_websocket_timeout,
)

from ._common import _DATASET_PATH, _PROJECT_ROOT

# ---------------------------------------------------------------------------
# Candidate and evaluator model catalog
# ---------------------------------------------------------------------------

# Candidate models to benchmark (name + provider).
CANDIDATE_MODELS = [
    {"name": "voxtrol-mini", "provider": "aws"},
    {"name": "gemma4_31b_it", "provider": "asu-air"},
    {"name": "gpt5_6_luna", "provider": "openai"},
    {"name": "gpt5_6_sol", "provider": "openai"},
    {"name": "geminiflash3_7", "provider": "gcp-deepmind"},
    {"name": "claude4_5_haiku", "provider": "aws"},
    {"name": "gpt5_4_nano", "provider": "openai"},
]

# Judge/evaluator model used by accuracy_with_reference.
EVALUATOR_MODEL = {"name": "gemma4_31b_it", "provider": "asu-air"}

# Fallback judge/evaluator model. Used when the candidate model being scored is
# Gemma, so that Gemma is never used to judge itself. Selected up front (not on
# error) based on the candidate model.
FALLBACK_EVALUATOR_MODEL = {"name": "gpt5_6_luna", "provider": "openai"}

ASURITE = "ethaie-datascience"
THRESHOLD = 0.5

# How long to wait between finalize polls, and the max number of polls.
POLL_INTERVAL_SECONDS = 10
MAX_POLLS = 180  # ~30 minutes at 10s intervals
MIN_POLL_INTERVAL_SECONDS = 0.1
MAX_POLL_INTERVAL_SECONDS = 600.0
MIN_POLLS = 1
MAX_ALLOWED_POLLS = 10_000

DEFAULT_CONCURRENCY = 1
MAX_CONCURRENCY = 16
DEFAULT_RETRY_COUNT = 0
MAX_RETRY_COUNT = 3


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _same_model(left: dict, right: dict) -> bool:
    """Compare model identities by both provider and model name."""
    return (
        left.get("name") == right.get("name")
        and left.get("provider") == right.get("provider")
    )


def _select_evaluator(
    candidate_model: dict,
    evaluator_model: dict | None = None,
) -> dict:
    """Pick a judge without allowing a model to grade itself."""
    selected = dict(evaluator_model or EVALUATOR_MODEL)
    if _same_model(candidate_model, selected):
        fallback = FALLBACK_EVALUATOR_MODEL
        if _same_model(candidate_model, fallback):
            fallback = EVALUATOR_MODEL
        if _same_model(candidate_model, fallback):
            raise ValueError("No evaluator model distinct from the candidate model is configured")
        return dict(fallback)
    return selected


def _select_candidate_models(
    model_name: str | None = None,
    model_provider: str | None = None,
) -> list[dict]:
    """Resolve optional CLI model filters while preserving default ordering."""
    if model_name is None and model_provider is None:
        return [dict(model) for model in CANDIDATE_MODELS]

    if model_name is not None and not model_name.strip():
        raise ValueError("candidate model name must not be empty")
    if model_provider is not None and not model_provider.strip():
        raise ValueError("candidate model provider must not be empty")

    if model_name is None:
        selected = [
            dict(model)
            for model in CANDIDATE_MODELS
            if model["provider"] == model_provider
        ]
        if not selected:
            raise ValueError(f"No default candidate models use provider {model_provider!r}")
        return selected

    if model_provider is None:
        selected = [
            dict(model) for model in CANDIDATE_MODELS if model["name"] == model_name
        ]
        if len(selected) == 1:
            return selected
        if not selected:
            raise ValueError(
                "--candidate-provider is required when selecting a non-default candidate model"
            )
        raise ValueError(
            "--candidate-provider is required when a candidate name has multiple providers"
        )

    return [{"name": model_name.strip(), "provider": model_provider.strip()}]


def _resolve_evaluator_model(
    model_name: str | None = None,
    model_provider: str | None = None,
) -> dict:
    """Resolve evaluator CLI overrides against the configured judge default."""
    if model_name is not None and not model_name.strip():
        raise ValueError("evaluator model name must not be empty")
    if model_provider is not None and not model_provider.strip():
        raise ValueError("evaluator model provider must not be empty")
    return {
        "name": (model_name or EVALUATOR_MODEL["name"]).strip(),
        "provider": (model_provider or EVALUATOR_MODEL["provider"]).strip(),
    }


# ---------------------------------------------------------------------------
# Bound validators
# ---------------------------------------------------------------------------

def validate_poll_interval_seconds(value: float) -> float:
    """Validate a finite, positive, bounded evaluator poll interval."""
    if isinstance(value, bool):
        raise ValueError("poll interval must be a finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("poll interval must be a finite number") from exc
    if not normalized == normalized or normalized in (float("inf"), float("-inf")):
        raise ValueError("poll interval must be a finite number")
    if normalized < MIN_POLL_INTERVAL_SECONDS:
        raise ValueError(
            f"poll interval must be at least {MIN_POLL_INTERVAL_SECONDS:g} seconds"
        )
    if normalized > MAX_POLL_INTERVAL_SECONDS:
        raise ValueError(
            f"poll interval must not exceed {MAX_POLL_INTERVAL_SECONDS:g} seconds"
        )
    return normalized


def validate_max_polls(value: int) -> int:
    """Validate a positive, bounded evaluator poll count."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max polls must be an integer")
    if value < MIN_POLLS:
        raise ValueError(f"max polls must be at least {MIN_POLLS}")
    if value > MAX_ALLOWED_POLLS:
        raise ValueError(f"max polls must not exceed {MAX_ALLOWED_POLLS}")
    return value


def build_execution_policy(
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    retry_count: int = DEFAULT_RETRY_COUNT,
) -> dict:
    """Validate and serialize the candidate execution policy.

    Sequential execution and no retries are deliberately the defaults.  A
    retry can repeat a request that already invoked a side-effecting tool, so
    callers must opt into it explicitly.  The bounded values keep accidental
    CLI settings from creating unbounded work.
    """
    for name, value in (("concurrency", concurrency), ("retry_count", retry_count)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0 or (name == "concurrency" and value == 0):
            raise ValueError(f"{name} must be greater than zero")

    if concurrency > MAX_CONCURRENCY:
        raise ValueError(f"concurrency must not exceed {MAX_CONCURRENCY}")
    if retry_count > MAX_RETRY_COUNT:
        raise ValueError(f"retry_count must not exceed {MAX_RETRY_COUNT}")

    return {
        "mode": "sequential" if concurrency == 1 else "bounded_thread_pool",
        "concurrency": concurrency,
        "retry_count": retry_count,
        "retry_enabled": retry_count > 0,
        "socket_sharing": False,
        "independent_client_per_scenario": concurrency > 1,
    }


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI parser while preserving legacy defaults."""
    parser = argparse.ArgumentParser(
        description="Run the tool-calling benchmark through candidate models and score accuracy.",
    )
    parser.add_argument("--dataset-path", type=str, default=str(_DATASET_PATH))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N scenarios (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_PROJECT_ROOT / "custom_benchmarks" / "results"),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--config-path", "--agent-config-path",
        type=Path,
        default=None,
        dest="config_path",
        help="Agentic credential file; defaults to AGENTIC_CONFIG_PATH or the checked-in convention.",
    )
    parser.add_argument(
        "--environment", "--agent-environment",
        default=None,
        dest="environment",
        help="Credential section; defaults to AGENTIC_ENV or DEV.",
    )
    parser.add_argument(
        "--timeout", "--websocket-timeout",
        type=validate_websocket_timeout,
        default=DEFAULT_WEBSOCKET_TIMEOUT_SECONDS,
        dest="timeout",
        help="Per-request WebSocket timeout in seconds (bounded by the agentic client).",
    )
    parser.add_argument(
        "--poll-interval", "--evaluator-poll-interval",
        type=validate_poll_interval_seconds,
        default=POLL_INTERVAL_SECONDS,
        dest="poll_interval_seconds",
        help=(
            "Seconds between evaluator finalize polls; must be between "
            f"{MIN_POLL_INTERVAL_SECONDS:g} and {MAX_POLL_INTERVAL_SECONDS:g}."
        ),
    )
    parser.add_argument(
        "--max-polls", "--evaluator-max-polls",
        type=lambda value: validate_max_polls(int(value)),
        default=MAX_POLLS,
        dest="max_polls",
        help=f"Maximum evaluator finalize polls (1-{MAX_ALLOWED_POLLS}).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=(
            "Maximum independent candidate WebSocket clients to run concurrently; "
            "default 1 preserves sequential execution."
        ),
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help=(
            "Explicitly enable up to this many retries for incomplete/failed traces; "
            "default 0 avoids repeating potentially side-effecting tool calls."
        ),
    )
    parser.add_argument(
        "--raw-observations", "--raw-output",
        action="store_true",
        dest="raw_observations",
        help="Write the optional redacted observations JSON artifact.",
    )
    parser.add_argument(
        "--raw-frames", "--include-raw-frames",
        action="store_true",
        dest="raw_frames",
        help="Retain redacted raw WebSocket frames in collected observations.",
    )
    parser.add_argument(
        "--dry-run",
        "--validation-only",
        action="store_true",
        dest="dry_run",
        help=(
            "Validate the selected dataset and print redacted agentic requests "
            "without loading credentials or opening a WebSocket."
        ),
    )
    parser.add_argument(
        "--no-write-ddb",
        "--no-persist",
        action="store_false",
        dest="write_ddb",
        help=(
            "Disable the per-model ethai_stats DynamoDB write for a local-only "
            "run. Persistence is enabled by default so runs are captured in the "
            "DynamoDB infrastructure; local CSV/XLSX/JSON artifacts are written "
            "either way."
        ),
    )
    parser.set_defaults(write_ddb=True)
    parser.add_argument(
        "--write-data-mart",
        action="store_true",
        dest="write_data_mart",
        help=(
            "After the DynamoDB writes, pull each model's run_id back from "
            "DynamoDB and append the stats + query_processor payloads to the "
            "Redshift data mart (aidwnp.ai_data_science.pre_release_*). Off by "
            "default; requires psycopg2 and data-mart credentials. Implies the "
            "DynamoDB write (ignored with --no-write-ddb)."
        ),
    )
    parser.add_argument(
        "--data-mart-env",
        default=None,
        dest="data_mart_env",
        help=(
            "Data-mart target environment for --write-data-mart: NONPROD "
            "(default) or PROD. Falls back to the DATA_MART_ENV variable."
        ),
    )
    parser.add_argument("--candidate-model", default=None)
    parser.add_argument("--candidate-provider", default=None)
    parser.add_argument("--evaluator-model", default=None)
    parser.add_argument("--evaluator-provider", default=None)
    return parser
