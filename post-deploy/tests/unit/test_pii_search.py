"""Unit tests for PiiSearchMetric."""

import pandas as pd
import pytest

from post_deploy.core.metric import MetricContext
from post_deploy.metrics.pii_search import PiiSearchMetric

# Skip all tests in this module if presidio is not installed
presidio = pytest.importorskip("presidio_analyzer")


@pytest.fixture
def sample_df():
    """Create a DataFrame with PII and non-PII text."""
    return pd.DataFrame({
        "query_raw": [
            "My email is jane@example.com and phone is 212-555-5555.",
            "What is the weather today?",
            "Contact Alex at alex@university.edu or visit example.edu.",
        ],
        "response_raw": [
            "Here is the information you requested.",
            "The weather is sunny today.",
            "Please check our website for details.",
        ],
    })


@pytest.fixture
def context():
    """Create a MetricContext with standard column mappings."""
    return MetricContext(columns={"query": "query_raw", "response": "response_raw"})


class TestPiiSearchMetric:
    """Tests for the PiiSearchMetric class."""

    def test_basic_properties(self):
        """Test metric name and version."""
        metric = PiiSearchMetric(config={"target_columns": ["query"]})
        assert metric.name == "pii_search"
        assert metric.version == "1.0.0"
        assert metric.required_columns == ["query"]

    def test_validate_config_missing_columns(self):
        """Validate raises on empty target_columns."""
        metric = PiiSearchMetric(config={})
        with pytest.raises(ValueError, match="target_columns"):
            metric.validate_config({"target_columns": []})

    def test_validate_config_missing_entities(self):
        """Validate raises on empty entity_types."""
        metric = PiiSearchMetric(config={})
        with pytest.raises(ValueError, match="entity_types"):
            metric.validate_config({"target_columns": ["query"], "entity_types": []})

    def test_validate_config_valid(self):
        """Validate passes with proper config."""
        config = {"target_columns": ["query"], "entity_types": ["EMAIL_ADDRESS"]}
        metric = PiiSearchMetric(config=config)
        metric.validate_config(config)

    def test_process_detects_pii(self, sample_df, context):
        """Test that PII is detected in text."""
        config = {
            "target_columns": ["query"],
            "entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
        }
        metric = PiiSearchMetric(config=config)
        result = metric.process(sample_df, context)

        # Check output columns exist
        assert "pii_query_any_found" in result.columns
        assert "pii_query_found_all" in result.columns
        assert "pii_query_found_distinct" in result.columns

        # Row 0: has email and phone
        assert result.iloc[0]["pii_query_any_found"] == True
        assert len(result.iloc[0]["pii_query_found_all"]) > 0

        # Row 1: no PII
        assert result.iloc[1]["pii_query_any_found"] == False
        assert result.iloc[1]["pii_query_found_all"] == []

    def test_process_no_pii_in_clean_text(self, context):
        """Test that clean text returns no PII findings."""
        df = pd.DataFrame({"query_raw": ["Hello, how are you?", "Tell me about Python."]})
        config = {"target_columns": ["query"]}
        metric = PiiSearchMetric(config=config)
        result = metric.process(df, context)

        assert all(result["pii_query_any_found"] == False)  # noqa: E712

    def test_verbose_mode_adds_entity_columns(self, sample_df, context):
        """Test that verbose mode adds per-entity-type columns."""
        config = {
            "target_columns": ["query"],
            "entity_types": ["EMAIL_ADDRESS", "PHONE_NUMBER"],
            "verbose": True,
        }
        metric = PiiSearchMetric(config=config)
        result = metric.process(sample_df, context)

        # Row 0 should have email entity detail
        row0 = result.iloc[0]
        # Check that at least one NE column exists for the found entities
        ne_cols = [c for c in result.columns if c.startswith("pii_query_ne_")]
        assert len(ne_cols) > 0

    def test_non_verbose_mode(self, sample_df, context):
        """Test that non-verbose mode omits per-entity-type columns."""
        config = {
            "target_columns": ["query"],
            "entity_types": ["EMAIL_ADDRESS"],
            "verbose": False,
        }
        metric = PiiSearchMetric(config=config)
        result = metric.process(sample_df, context)

        # Should not have NE detail columns
        ne_cols = [c for c in result.columns if c.startswith("pii_query_ne_")]
        assert len(ne_cols) == 0

    def test_empty_dataframe(self, context):
        """Test processing an empty DataFrame."""
        df = pd.DataFrame({"query_raw": pd.Series([], dtype=str)})
        config = {"target_columns": ["query"]}
        metric = PiiSearchMetric(config=config)
        result = metric.process(df, context)

        assert "pii_query_any_found" in result.columns
        assert len(result) == 0
