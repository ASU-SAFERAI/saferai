"""Pydantic configuration models for the post-deploy pipeline."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class OutputFormat(str, Enum):
    """Supported output DataFrame formats."""

    LONG = "long"
    WIDE = "wide"


class InputConfig(BaseModel):
    """Configuration for the pipeline's input source."""

    type: str = Field(default="local", description="Input source type: 'local', 's3', or 'database'.")
    paths: list[str] = Field(default_factory=list, description="File paths or S3 prefixes.")
    columns: dict[str, str] = Field(
        default_factory=lambda: {"query": "query_raw", "response": "response_raw"},
        description="Mapping of logical column names to actual DataFrame column names.",
    )

    # Database-specific
    schema_name: str | None = Field(default=None, description="DB schema for database input.")
    table_name: str | None = Field(default=None, description="DB table for database input.")

    # S3-specific
    bucket: str | None = Field(default=None, description="S3 bucket name.")
    prefix: str | None = Field(default=None, description="S3 prefix for file discovery.")


class OutputConfig(BaseModel):
    """Configuration for the pipeline's output destination."""

    type: str = Field(default="local", description="Output type: 'local', 's3', or 'database'.")
    dir: str = Field(default="outputs", description="Local output directory.")
    format: OutputFormat = Field(default=OutputFormat.LONG, description="Output format: 'long' or 'wide'.")

    # Database-specific
    schema_name: str | None = Field(default=None, description="DB schema for database output.")
    table_name: str | None = Field(default=None, description="DB table for database output.")

    # S3-specific
    bucket: str | None = Field(default=None, description="S3 bucket name.")
    prefix: str | None = Field(default=None, description="S3 prefix for output.")


class MetricConfig(BaseModel):
    """Configuration for a single metric within the pipeline."""

    name: str = Field(description="Metric name matching a registered metric (e.g., 'keyword_search').")
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific configuration passed to validate_config and __init__.",
    )
    enabled: bool = Field(default=True, description="Whether this metric is active.")


class PostProcessConfig(BaseModel):
    """Configuration for the post-processing step (wide-to-long melt)."""

    enabled: bool = Field(default=True, description="Whether to run post-processing.")
    id_cols: list[str] = Field(
        default_factory=list,
        description="Columns to keep as identifiers during melt. Only present columns are used.",
    )
    drop_cols: list[str] = Field(
        default_factory=list,
        description="Columns to drop before melting (e.g., raw text columns).",
    )
    version_column: str = Field(default="engine_version", description="Column name for the engine version.")
    run_day_column: str = Field(default="run_day", description="Column name for the processing date.")


class PipelineConfig(BaseModel):
    """
    Top-level pipeline configuration.

    Can be loaded from a YAML file or constructed programmatically.
    """

    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    metrics: list[MetricConfig] = Field(default_factory=list)
    post_process: PostProcessConfig = Field(default_factory=PostProcessConfig)

    # Optional metadata
    preset: str | None = Field(default=None, description="Name of a preset to load defaults from.")
    version: str | None = Field(default=None, description="Pipeline config version for tracking.")

    @field_validator("metrics")
    @classmethod
    def at_least_one_metric(cls, v: list[MetricConfig]) -> list[MetricConfig]:
        """Warn (but don't fail) if no metrics are configured."""
        # We allow empty metrics list for validation/testing purposes
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """
        Load a PipelineConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A validated PipelineConfig instance.

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
            ValueError: If the YAML content is invalid.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        with open(filepath) as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping at top level, got {type(raw).__name__}")

        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        """
        Serialize this config to a YAML file.

        Args:
            path: Destination file path.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump(mode="json")
        with open(filepath, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def get_enabled_metrics(self) -> list[MetricConfig]:
        """Return only metrics where enabled=True."""
        return [m for m in self.metrics if m.enabled]
