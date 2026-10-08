"""Benchmark orchestration: wire configuration, execution, scoring, and reporting.

This module is deliberately thin. Each concern lives in its own module
(:mod:`config`, :mod:`data`, :mod:`execution`, :mod:`evaluation`,
:mod:`reporting`); ``main`` composes them into the end-to-end run. The names
imported here are also the monkeypatch surface used by the CLI facade and tests.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import pandas as pd

from pre_deploy.input import EvalDataset
from pre_deploy.query_processor import AWSEnvironment
from pre_deploy.query_processor.alert_manager import AlertManager
from custom_benchmarks.tool_calling.agentic.client import AgenticWebSocketClient
from custom_benchmarks.tool_calling.agentic.config import (
    load_credentials,
    redact_sensitive_text,
)
from custom_benchmarks.tool_calling.models import ToolCallObservation

from ._common import configure_logging, logger
from .config import (
    EVALUATOR_MODEL,
    MAX_POLLS,
    POLL_INTERVAL_SECONDS,
    THRESHOLD,
    _resolve_evaluator_model,
    _same_model,
    _select_candidate_models,
    _select_evaluator,
    build_argument_parser,
    build_execution_policy,
)
from .data import (
    build_dry_run_report,
    _print_dry_run_report,
    build_trace_eval_dataset,
    load_scenarios,
)
from .evaluation import score_accuracy
from .evaluation import (
    finalize_tool_call_failure_modes,
    sleep,
    start_tool_call_failure_modes,
)
from .data_mart import write_benchmark_run_to_data_mart
from .execution import collect_candidate_responses
from .persistence import persist_model_stats
from .reporting import build_report, build_summary
from .artifacts import (
    build_dataset_provenance,
    build_registry_provenance,
    build_run_manifest,
    write_outputs,
    _utc_timestamp,
)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    scenarios = load_scenarios(Path(args.dataset_path), limit=args.limit)
    dataset_provenance = build_dataset_provenance(args.dataset_path, len(scenarios))
    benchmark_run_id = str(uuid.uuid4())
    run_started_at = _utc_timestamp()
    logger.info("Loaded %d scenarios from %s", len(scenarios), args.dataset_path)

    try:
        execution_policy = build_execution_policy(
            concurrency=args.concurrency,
            retry_count=args.retry_count,
        )
        candidate_models = _select_candidate_models(
            args.candidate_model,
            args.candidate_provider,
        )
        if args.dry_run:
            _print_dry_run_report(
                build_dry_run_report(
                    scenarios,
                    candidate_models,
                    dataset_path=Path(args.dataset_path),
                )
            )
            return

        evaluator_model_config = _resolve_evaluator_model(
            args.evaluator_model,
            args.evaluator_provider,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        message = redact_sensitive_text(str(exc))
        logger.error("Benchmark configuration failed: %s", message)
        raise SystemExit(2) from exc

    # Credential loading and client construction are run-level gates.  Keep
    # their failures outside the per-scenario loop: without an authenticated
    # client no candidate request can be attempted, so continuing would only
    # produce an ambiguous empty run.  Catch construction/configuration errors
    # broadly because configparser, filesystem, and TLS setup failures vary by
    # environment; redact the detail before logging it.
    try:
        agentic_credentials = load_credentials(
            config_path=args.config_path,
            environment=args.environment,
        )
        agentic_client = AgenticWebSocketClient(
            agentic_credentials,
            timeout=args.timeout,
        )
    except Exception as exc:
        message = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
        logger.error("Agentic configuration/authentication failed: %s", message)
        raise SystemExit(2) from exc

    logger.info(
        "Candidate execution policy: mode=%s concurrency=%d retries=%d "
        "socket_sharing=%s",
        execution_policy["mode"],
        execution_policy["concurrency"],
        execution_policy["retry_count"],
        execution_policy["socket_sharing"],
    )
    client_factory = (
        (lambda: AgenticWebSocketClient(agentic_credentials, timeout=args.timeout))
        if args.concurrency > 1
        else None
    )

    # The evaluator environment is created lazily by score_accuracy so dry-run
    # and mocked scoring paths do not require AWS credentials or network access.
    environment = None

    all_rows: list[dict] = []
    all_observations: list[ToolCallObservation] = []
    evaluator_models_used: list[dict[str, str]] = []

    # DynamoDB persistence is enabled by default so runs are captured in the
    # DynamoDB infrastructure. The environment/alert manager are created lazily
    # on first use so dry-run and --no-write-ddb paths need no AWS access. The
    # per-question query_processor items are written by the staged evaluator
    # itself; the only net-new write here is one ethai_stats item per model.
    write_ddb = getattr(args, "write_ddb", True)
    stats_environment: AWSEnvironment | None = None
    stats_alert_manager: AlertManager | None = None
    persisted_stats: list[dict[str, str]] = []

    total_models = len(candidate_models)
    logger.info(
        "Benchmark run %s starting: %d candidate model(s) x %d scenario(s)",
        benchmark_run_id, total_models, len(scenarios),
    )

    for model_index, model_info in enumerate(candidate_models, start=1):
        model_name = model_info["name"]
        logger.info(
            "=== Candidate model %d/%d: %s/%s ===",
            model_index, total_models, model_name, model_info["provider"],
        )
        observations: dict[str, ToolCallObservation] = {}
        eval_dataset: EvalDataset | None = None
        evaluator_model: dict | None = None
        try:
            observations = collect_candidate_responses(
                model_info,
                scenarios,
                agentic_client,
                include_raw_frames=args.raw_frames,
                concurrency=args.concurrency,
                retry_count=args.retry_count,
                client_factory=client_factory,
            )
            all_observations.extend(observations.values())
            logger.info(
                "Building evaluation dataset for %s/%s from %d observation(s)",
                model_name, model_info["provider"], len(observations),
            )
            eval_dataset = build_trace_eval_dataset(
                scenarios,
                observations,
                dataset_version_id=dataset_provenance["version"],
                metadata={
                    "execution_policy": json.dumps(
                        execution_policy,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            )

            evaluator_model = _select_evaluator(model_info, evaluator_model_config)
            if evaluator_model not in evaluator_models_used:
                evaluator_models_used.append(dict(evaluator_model))
            if not _same_model(evaluator_model, EVALUATOR_MODEL):
                logger.info(
                    "Using evaluator %s/%s for candidate %s/%s.",
                    evaluator_model["name"], evaluator_model["provider"],
                    model_info["name"], model_info["provider"],
                )

            logger.info(
                "Scoring %d scenario(s) for %s/%s with evaluator %s/%s",
                len(observations), model_name, model_info["provider"],
                evaluator_model["name"], evaluator_model["provider"],
            )
            scored = score_accuracy(
                eval_dataset,
                environment,
                evaluator_model,
                poll_interval_seconds=args.poll_interval_seconds,
                max_polls=args.max_polls,
            )
            logger.info(
                "Building report rows for %s/%s", model_name, model_info["provider"],
            )
            model_rows = build_report(
                model_name,
                scored,
                eval_dataset,
                model_provider=model_info["provider"],
                observations=observations,
                scenarios=scenarios,
                evaluator_model=evaluator_model,
            )
            all_rows.extend(model_rows)

            # Persist one ethai_stats summary item for this candidate model once
            # its metric computation is complete. Keyed by the evaluator run_id
            # carried on the MetricsResults so the stats row correlates with the
            # per-question query_processor items the staged evaluator already
            # wrote under the same run_id. Enabled by default; a failed write
            # follows the alert-and-re-raise convention via persist_model_stats.
            if write_ddb:
                model_run_id = scored.run_id
                model_summary_df = build_summary(pd.DataFrame(model_rows))
                summary_row = (
                    model_summary_df.iloc[0].to_dict()
                    if not model_summary_df.empty
                    else None
                )
                if stats_environment is None:
                    stats_environment = AWSEnvironment(
                        target_account_id=None, role_name=None
                    )
                    stats_alert_manager = AlertManager(
                        aws_environment=stats_environment
                    )
                persist_model_stats(
                    stats_environment,
                    stats_alert_manager,
                    run_id=model_run_id,
                    model_name=model_name,
                    model_provider=model_info["provider"],
                    evaluator_model_name=evaluator_model["name"],
                    evaluator_model_provider=evaluator_model["provider"],
                    threshold=(
                        scored.dataset_info.get("threshold", THRESHOLD)
                        if isinstance(scored.dataset_info, dict)
                        else THRESHOLD
                    ),
                    metrics_results=scored,
                    summary_row=summary_row,
                )
                persisted_stats.append(
                    {
                        "model": model_name,
                        "provider": model_info["provider"],
                        "run_id": model_run_id,
                    }
                )

            logger.info(
                "Finished candidate model %d/%d (%s/%s): %d report row(s)",
                model_index, total_models, model_name, model_info["provider"],
                len(model_rows),
            )
        except Exception as exc:
            failure = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
            logger.error(
                "Benchmark failed for model %s/%s; continuing with remaining models: %s",
                model_info.get("name"),
                model_info.get("provider"),
                failure,
            )
            # Keep one visible row per attempted scenario when evaluation
            # fails. Candidate observations must survive a judge timeout,
            # unavailable evaluator, or any other evaluator lifecycle error.
            if observations:
                if eval_dataset is None:
                    eval_dataset = build_trace_eval_dataset(
                        scenarios,
                        observations,
                        dataset_version_id=dataset_provenance["version"],
                        metadata={},
                    )
                all_rows.extend(
                    build_report(
                        model_name,
                        None,
                        eval_dataset,
                        model_provider=model_info["provider"],
                        observations=observations,
                        scenarios=scenarios,
                        evaluator_model=evaluator_model,
                        evaluator_status="failed",
                        evaluator_failure=failure,
                    )
                )

    # A report row without an observation is not a candidate result.  Per-
    # scenario failures do count as reportable results because they are
    # retained as explicit observations and detail rows.
    if not all_rows:
        logger.error("No reportable candidate results were produced. Nothing to write.")
        sys.exit(1)

    logger.info(
        "All candidate models processed: %d report row(s) across %d model(s); "
        "building summary",
        len(all_rows), total_models,
    )
    report_df = pd.DataFrame(all_rows)
    summary_df = build_summary(report_df)

    print("\n=== Per-model accuracy summary ===")
    print(summary_df.to_string(index=False))

    run_manifest = build_run_manifest(
        run_id=benchmark_run_id,
        dataset=dataset_provenance,
        candidate_models=candidate_models,
        evaluator_model=evaluator_model_config,
        timeout=args.timeout,
        execution_policy=execution_policy,
        started_at=run_started_at,
        ended_at=_utc_timestamp(),
        registry=build_registry_provenance(),
        websocket_config={
            "config_path": str(args.config_path) if args.config_path else None,
            "environment": args.environment,
            "authenticated_ws_url": redact_sensitive_text(
                agentic_credentials.authenticated_ws_url
            ),
        },
        evaluator_models=evaluator_models_used,
        evaluator_poll_interval_seconds=args.poll_interval_seconds,
        evaluator_max_polls=args.max_polls,
        observation_count=len(all_observations),
        raw_frames=args.raw_frames,
        raw_observations=args.raw_observations,
    )
    # Record how the run was persisted so a stats row can be traced back to the
    # local artifacts. ``stats_table`` is resolved via the same AWSEnvironment
    # attribute used for the write; when persistence is disabled the env var
    # default is reported without constructing an AWS session.
    run_manifest["persistence"] = {
        "write_ddb": write_ddb,
        "stats_table": (
            stats_environment.stats_table
            if stats_environment is not None
            else os.environ.get("stats_table", "ethai_stats_dev")
        ),
        "query_processor_source": "staged_tool_call_failure_modes_eval_phase",
        "persisted_stats": persisted_stats,
    }

    # Optional Redshift data-mart write. Pull each model's run_id back from
    # DynamoDB and append the stats + query_processor payloads to the shared
    # data-mart tables other EthAI metrics use. Off by default; requires the
    # DynamoDB write to have produced run_ids and the psycopg2 driver.
    write_data_mart = getattr(args, "write_data_mart", False)
    data_mart_result: dict | None = None
    if write_data_mart:
        data_mart_run_ids = [entry["run_id"] for entry in persisted_stats]
        if not write_ddb:
            logger.warning(
                "--write-data-mart requested with --no-write-ddb; nothing was "
                "written to DynamoDB, so the data-mart write is skipped."
            )
        elif not data_mart_run_ids:
            logger.warning(
                "--write-data-mart requested but no run_ids were persisted to "
                "DynamoDB; skipping the data-mart write."
            )
        else:
            try:
                data_mart_result = write_benchmark_run_to_data_mart(
                    data_mart_run_ids,
                    environment=stats_environment,
                    data_mart_environment=args.data_mart_env,
                    config_path=args.config_path,
                )
            except Exception as exc:
                # The data-mart write is a downstream, opt-in step. A failure
                # must not discard the DynamoDB writes or local artifacts that
                # already succeeded; surface it and continue to reporting.
                failure = redact_sensitive_text(f"{type(exc).__name__}: {exc}")
                logger.error("Data-mart write failed: %s", failure)
                data_mart_result = {"error": failure}
    run_manifest["data_mart"] = {
        "write_data_mart": write_data_mart,
        "environment": args.data_mart_env or "NONPROD" if write_data_mart else None,
        "rows_written": data_mart_result,
    }

    logger.info("Writing artifacts to %s", args.output_dir)
    output_paths = write_outputs(
        report_df,
        summary_df,
        Path(args.output_dir),
        observations=all_observations,
        run_manifest=run_manifest,
        include_raw_observations=args.raw_observations,
    )
    logger.info(
        "Benchmark run %s complete. Detail: %s | Summary: %s",
        benchmark_run_id,
        output_paths.get("detail_csv"),
        output_paths.get("summary_csv"),
    )


if __name__ == "__main__":
    main()
