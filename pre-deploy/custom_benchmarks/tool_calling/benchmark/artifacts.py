"""Provenance, run manifest, report-frame preparation, and artifact writing.

Owns everything about turning finished report frames into files on disk: the
stable CSV/XLSX schema boundary, dataset/registry/run provenance, and the
manifest plus optional redacted observation payloads.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from custom_benchmarks.tool_calling.agentic.config import redact_sensitive_data
from custom_benchmarks.tool_calling.models import ToolCallObservation
from custom_benchmarks.tool_calling.registry import TOOL_REGISTRY
from custom_benchmarks.tool_calling.serialization import canonical_json

from ._common import logger
from .config import (
    MAX_POLLS,
    POLL_INTERVAL_SECONDS,
    THRESHOLD,
    validate_max_polls,
    validate_poll_interval_seconds,
)
from .reporting import audit_columns_for

REPORT_SCHEMA_VERSION = 1

# Report columns containing structured values are serialized before writing so
# CSV consumers never receive Python reprs such as ``{'key': 'value'}``.
_REPORT_JSON_COLUMNS = frozenset(
    {
        "expected_tools",
        "enabled_tools",
        "tool_catalog",
        "scenario_metadata",
        "expected_calls",
        "observed_calls",
        "tool_outputs",
        "request",
        "stream_summary",
        "eos_metadata",
        "errors",
        "argument_parse_errors",
    }
)


def _json_report_value(value: Any) -> str:
    """Return a valid JSON document for a nested report field."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "null"
    if isinstance(value, str):
        try:
            json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return canonical_json(value)
        return value
    return canonical_json(value)


def _prepare_report_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Copy a report frame and apply the stable CSV/report schema boundary."""
    prepared = frame.copy()
    if "report_schema_version" not in prepared.columns:
        prepared.insert(0, "report_schema_version", REPORT_SCHEMA_VERSION)
    for column in _REPORT_JSON_COLUMNS.intersection(prepared.columns):
        prepared[column] = prepared[column].map(_json_report_value)
    return prepared


def _artifact_id() -> str:
    """Create a readable timestamp plus unique suffix for report artifacts."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for run provenance."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    """Hash a dataset or other provenance file without loading it at once."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_provenance(dataset_path: Path | str, scenario_count: int) -> dict[str, Any]:
    """Build stable, non-secret provenance for the selected dataset."""
    path = Path(dataset_path).expanduser().resolve()
    version = _sha256_file(path)
    return {
        "path": str(path),
        "version": version,
        "sha256": version,
        "scenario_count": scenario_count,
        "validation": "passed",
    }


def build_registry_provenance() -> dict[str, Any]:
    """Hash the canonical registry projection used by the benchmark."""
    serialized = canonical_json(TOOL_REGISTRY).encode("utf-8")
    return {
        "version": hashlib.sha256(serialized).hexdigest(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "tool_count": len(TOOL_REGISTRY),
        "tool_names": list(TOOL_REGISTRY),
    }


def build_run_manifest(
    *,
    run_id: str,
    dataset: Mapping[str, Any],
    candidate_models: Iterable[Mapping[str, Any]],
    evaluator_model: Mapping[str, Any] | None,
    timeout: float,
    execution_policy: Mapping[str, Any],
    started_at: str,
    ended_at: str | None = None,
    registry: Mapping[str, Any] | None = None,
    websocket_config: Mapping[str, Any] | None = None,
    evaluator_models: Iterable[Mapping[str, Any]] | None = None,
    evaluator_poll_interval_seconds: float | None = None,
    evaluator_max_polls: int | None = None,
    observation_count: int = 0,
    raw_frames: bool = False,
    raw_observations: bool = False,
) -> dict[str, Any]:
    """Create the run-level provenance document before artifact linking."""
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if not isinstance(raw_frames, bool) or not isinstance(raw_observations, bool):
        raise TypeError("raw_frames and raw_observations must be booleans")
    poll_interval = validate_poll_interval_seconds(
        POLL_INTERVAL_SECONDS
        if evaluator_poll_interval_seconds is None
        else evaluator_poll_interval_seconds
    )
    poll_count = validate_max_polls(
        MAX_POLLS if evaluator_max_polls is None else evaluator_max_polls
    )
    manifest = {
        "manifest_schema_version": 1,
        "run_id": run_id,
        "benchmark": "tool_calling",
        "started_at": started_at,
        "ended_at": ended_at,
        "dataset": dict(dataset),
        "candidate_models": [dict(model) for model in candidate_models],
        "evaluator_model": dict(evaluator_model) if evaluator_model else None,
        "evaluator_models": [
            dict(model) for model in (evaluator_models or ([evaluator_model] if evaluator_model else []))
        ],
        "metric": {
            "name": "tool_call_failure_modes",
            "threshold": THRESHOLD,
        },
        "evaluator_polling": {
            "interval_seconds": poll_interval,
            "max_polls": poll_count,
        },
        "websocket": {
            "timeout_seconds": timeout,
            "config": redact_sensitive_data(dict(websocket_config or {})),
        },
        "execution_policy": dict(execution_policy),
        "registry": dict(registry or build_registry_provenance()),
        "observation_count": observation_count,
        "raw_frames": raw_frames,
        "raw_observations": raw_observations,
        "artifacts": {},
    }
    return redact_sensitive_data(manifest)


def _json_document(value: Any) -> str:
    """Render a readable JSON document after canonicalizing unusual values."""
    return json.dumps(
        json.loads(canonical_json(value)),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )


def _observation_payloads(
    observations: Iterable[ToolCallObservation],
) -> list[dict[str, Any]]:
    """Serialize observations while preserving their redacted raw traces."""
    return [
        redact_sensitive_data(observation.to_dict())
        for observation in observations
    ]


def write_outputs(
    report_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    observations: Iterable[ToolCallObservation] | None = None,
    run_manifest: Mapping[str, Any] | None = None,
    include_raw_observations: bool = False,
) -> dict[str, Path | None]:
    """Write reports, a provenance manifest, and optional raw observations.

    Raw observations are opt-in.  Observation objects are already redacted at
    collection time, and this boundary redacts them again before writing so
    request payloads, errors, and optional raw frames cannot leak credentials.
    """
    if not isinstance(include_raw_observations, bool):
        raise TypeError("include_raw_observations must be a boolean")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_id = _artifact_id()
    # The detail artifact is the pared-down audit view: candidate/scenario
    # identity, dataset metadata, golden truth, observed tool calls, and the
    # judge verdict. The full report frame is still used to build the summary
    # upstream; only the on-disk detail output is projected. ``run_id`` is
    # preserved (needed to correlate with the manifest), so the audit projection
    # is applied before the schema-version column is inserted.
    detail_frame = _prepare_report_frame(report_df[audit_columns_for(report_df.columns)])
    summary_frame = _prepare_report_frame(summary_df)

    detail_csv = output_dir / f"tool_calling_benchmark_detail_{artifact_id}.csv"
    summary_csv = output_dir / f"tool_calling_benchmark_summary_{artifact_id}.csv"
    detail_frame.to_csv(detail_csv, index=False)
    summary_frame.to_csv(summary_csv, index=False)
    logger.info("Wrote %s", detail_csv)
    logger.info("Wrote %s", summary_csv)

    excel_path: Path | None = output_dir / f"tool_calling_benchmark_{artifact_id}.xlsx"
    # Excel workbook with both sheets, if openpyxl is available.  This is
    # intentionally attempted only after both required CSV writes succeed.
    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_frame.to_excel(writer, sheet_name="summary", index=False)
            detail_frame.to_excel(writer, sheet_name="detail", index=False)
        logger.info("Wrote %s", excel_path)
    except ImportError:
        logger.warning("openpyxl not installed; skipping Excel output (CSV files written).")
        excel_path = None

    manifest = dict(run_manifest or {})
    manifest.setdefault("manifest_schema_version", 1)
    manifest.setdefault("run_id", artifact_id)
    manifest.setdefault("benchmark", "tool_calling")
    manifest.setdefault("started_at", _utc_timestamp())
    manifest.setdefault("ended_at", _utc_timestamp())
    manifest["artifacts"] = {
        **dict(manifest.get("artifacts") or {}),
        "detail_csv": str(detail_csv),
        "summary_csv": str(summary_csv),
        "xlsx": str(excel_path) if excel_path is not None else None,
    }
    manifest = redact_sensitive_data(manifest)
    manifest_path = output_dir / f"tool_calling_benchmark_manifest_{artifact_id}.json"

    observation_list = list(observations or [])
    observation_path: Path | None = None
    if include_raw_observations:
        observation_path = output_dir / f"tool_calling_benchmark_observations_{artifact_id}.json"
        manifest["artifacts"]["observations_json"] = str(observation_path)
        raw_payload = {
            "manifest": manifest,
            "observations": _observation_payloads(observation_list),
        }
        observation_path.write_text(_json_document(raw_payload), encoding="utf-8")
        logger.info("Wrote %s", observation_path)

    manifest_path.write_text(_json_document(manifest), encoding="utf-8")
    logger.info("Wrote %s", manifest_path)

    return {
        "detail_csv": detail_csv,
        "summary_csv": summary_csv,
        "xlsx": excel_path,
        "manifest": manifest_path,
        "observations_json": observation_path,
    }
