"""Unit tests for PipelineConfig and related Pydantic models."""

import tempfile
from pathlib import Path

import pytest
import yaml

from post_deploy.core.config import (
    InputConfig,
    MetricConfig,
    OutputConfig,
    OutputFormat,
    PipelineConfig,
    PostProcessConfig,
)


class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_defaults(self):
        """Test that defaults are sensible."""
        config = PipelineConfig()
        assert config.input.type == "local"
        assert config.output.format == OutputFormat.LONG
        assert config.post_process.enabled is True
        assert config.metrics == []

    def test_get_enabled_metrics(self):
        """Test filtering of enabled metrics."""
        config = PipelineConfig(
            metrics=[
                MetricConfig(name="a", enabled=True),
                MetricConfig(name="b", enabled=False),
                MetricConfig(name="c", enabled=True),
            ]
        )
        enabled = config.get_enabled_metrics()
        assert len(enabled) == 2
        assert [m.name for m in enabled] == ["a", "c"]

    def test_from_yaml(self):
        """Test loading config from a YAML file."""
        yaml_content = {
            "input": {
                "type": "local",
                "paths": ["data/test.csv"],
                "columns": {"query": "q_col", "response": "r_col"},
            },
            "metrics": [
                {"name": "keyword_search", "config": {"target_columns": ["query"]}},
            ],
            "output": {"type": "local", "dir": "out", "format": "wide"},
            "post_process": {"enabled": False},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(yaml_content, f)
            tmp_path = f.name

        config = PipelineConfig.from_yaml(tmp_path)
        assert config.input.type == "local"
        assert config.input.paths == ["data/test.csv"]
        assert config.input.columns == {"query": "q_col", "response": "r_col"}
        assert len(config.metrics) == 1
        assert config.metrics[0].name == "keyword_search"
        assert config.output.format == OutputFormat.WIDE
        assert config.post_process.enabled is False

        Path(tmp_path).unlink()

    def test_from_yaml_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/path.yaml")

    def test_to_yaml(self):
        """Test serializing config to YAML."""
        config = PipelineConfig(
            input=InputConfig(type="local", paths=["test.csv"]),
            metrics=[MetricConfig(name="test_metric")],
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            tmp_path = f.name

        config.to_yaml(tmp_path)

        # Reload and verify
        loaded = PipelineConfig.from_yaml(tmp_path)
        assert loaded.input.paths == ["test.csv"]
        assert loaded.metrics[0].name == "test_metric"

        Path(tmp_path).unlink()


class TestInputConfig:
    """Tests for InputConfig."""

    def test_default_columns(self):
        """Test default column mappings."""
        cfg = InputConfig()
        assert cfg.columns == {"query": "query_raw", "response": "response_raw"}


class TestOutputConfig:
    """Tests for OutputConfig."""

    def test_format_enum(self):
        """Test output format enum values."""
        cfg = OutputConfig(format="long")
        assert cfg.format == OutputFormat.LONG

        cfg = OutputConfig(format="wide")
        assert cfg.format == OutputFormat.WIDE


class TestPostProcessConfig:
    """Tests for PostProcessConfig."""

    def test_defaults(self):
        """Test default values."""
        cfg = PostProcessConfig()
        assert cfg.enabled is True
        assert cfg.id_cols == []
        assert cfg.drop_cols == []
        assert cfg.version_column == "engine_version"
        assert cfg.run_day_column == "run_day"
