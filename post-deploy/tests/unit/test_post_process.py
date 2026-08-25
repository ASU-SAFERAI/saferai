"""Unit tests for post-processing utilities."""

import pandas as pd
import pytest

from post_deploy.core.config import OutputFormat, PostProcessConfig
from post_deploy.core.post_process import post_process_dataframe


@pytest.fixture
def wide_df():
    """Create a sample wide-format DataFrame."""
    return pd.DataFrame({
        "project_id": ["proj1", "proj2", "proj3"],
        "query_raw": ["hello", "world", "test"],
        "response_raw": ["hi", "there", "ok"],
        "metric_a": [0.9, 0.3, 0.7],
        "metric_b": [True, False, True],
    })


class TestPostProcessDataframe:
    """Tests for the post_process_dataframe function."""

    def test_long_format_basic(self, wide_df):
        """Test basic wide-to-long melt."""
        config = PostProcessConfig(
            enabled=True,
            id_cols=["project_id"],
            drop_cols=["query_raw", "response_raw"],
        )
        result = post_process_dataframe(wide_df, config, OutputFormat.LONG)

        assert "metric_name" in result.columns
        assert "metric_value" in result.columns
        assert "engine_version" in result.columns
        assert "run_day" in result.columns

        # 3 rows * 2 metric columns = 6 rows
        assert len(result) == 6
        assert set(result["metric_name"].unique()) == {"metric_a", "metric_b"}

    def test_wide_format_adds_metadata(self, wide_df):
        """Test wide format just adds metadata columns."""
        config = PostProcessConfig(enabled=True)
        result = post_process_dataframe(wide_df, config, OutputFormat.WIDE)

        # Should still have all original columns plus metadata
        assert "engine_version" in result.columns
        assert "run_day" in result.columns
        assert len(result) == 3  # Same row count

    def test_disabled_returns_unchanged(self, wide_df):
        """Test that disabled post-processing returns df unchanged."""
        config = PostProcessConfig(enabled=False)
        result = post_process_dataframe(wide_df, config, OutputFormat.LONG)

        assert result.equals(wide_df)

    def test_id_cols_only_existing(self, wide_df):
        """Test that non-existent id_cols are silently ignored."""
        config = PostProcessConfig(
            enabled=True,
            id_cols=["project_id", "nonexistent_col"],
            drop_cols=["query_raw", "response_raw"],
        )
        result = post_process_dataframe(wide_df, config, OutputFormat.LONG)

        # Should work without error, only using project_id
        assert "project_id" in result.columns
        assert "nonexistent_col" not in result.columns

    def test_no_metric_cols_raises(self):
        """Test that error is raised when no metric columns remain."""
        df = pd.DataFrame({"id_col": ["a", "b"], "text_col": ["hi", "bye"]})
        config = PostProcessConfig(
            enabled=True,
            id_cols=["id_col"],
            drop_cols=["text_col"],
        )
        with pytest.raises(ValueError, match="No metric columns"):
            post_process_dataframe(df, config, OutputFormat.LONG)

    def test_custom_column_names(self, wide_df):
        """Test custom version and run_day column names."""
        config = PostProcessConfig(
            enabled=True,
            id_cols=["project_id"],
            drop_cols=["query_raw", "response_raw"],
            version_column="safer_version",
            run_day_column="safer_run_day",
        )
        result = post_process_dataframe(wide_df, config, OutputFormat.LONG)

        assert "safer_version" in result.columns
        assert "safer_run_day" in result.columns
        assert "engine_version" not in result.columns
