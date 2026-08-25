"""Unit tests for LLMClassifierMetric."""

from __future__ import annotations

from typing import Dict
from unittest.mock import MagicMock

import pandas as pd
import pytest

from post_deploy.core.metric import MetricContext
from post_deploy.io.llm_client import BaseLLMClient
from post_deploy.metrics.llm_classifier import LLMClassifierMetric


class MockLLMClient(BaseLLMClient):
    """A mock LLM client that returns predefined responses."""

    def __init__(self, responses: Dict[int, str]):
        self._responses = responses

    def query(self, prompt: str) -> str:
        return "asking_a_question"

    def batch_query(self, prompts: Dict[int, str], max_concurrent: int = 4) -> Dict[int, str]:
        return {idx: self._responses.get(idx, "") for idx in prompts}


@pytest.fixture
def context():
    """Create a MetricContext with standard column mappings."""
    return MetricContext(columns={"query": "query_raw", "response": "response_raw"})


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame."""
    return pd.DataFrame({
        "query_raw": [
            "What is the refund policy?",
            "Hello there!",
            "How do I reset my password?",
        ],
        "response_raw": [
            "The refund policy allows returns within 30 days.",
            "I'm sorry, I can't answer that since it does not exist in the Knowledge Base.",
            "Could you please clarify your question?",
        ],
    })


class TestLLMClassifierMetric:
    """Tests for the LLMClassifierMetric class."""

    def test_basic_properties(self):
        """Test metric name and version."""
        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "template": "Classify: {text}",
                    "categories": ["asking_a_question", "other"],
                    "target": "query",
                    "prefix": "llm_q",
                }
            ],
            "client_instance": MockLLMClient({}),
        }
        metric = LLMClassifierMetric(config=config)
        assert metric.name == "llm_classifier"
        assert metric.version == "1.0.0"
        assert metric.required_columns == ["query"]

    def test_validate_config_missing_columns(self):
        """Validate raises on empty target_columns."""
        metric = LLMClassifierMetric(config={"client_instance": MockLLMClient({})})
        with pytest.raises(ValueError, match="target_columns"):
            metric.validate_config({"target_columns": [], "prompts": [{}], "client_instance": True})

    def test_validate_config_missing_prompts(self):
        """Validate raises when no prompt source provided."""
        metric = LLMClassifierMetric(config={"client_instance": MockLLMClient({})})
        with pytest.raises(ValueError, match="prompt source"):
            metric.validate_config({"target_columns": ["query"], "client_instance": True})

    def test_validate_config_missing_client(self):
        """Validate raises when no LLM client config provided."""
        metric = LLMClassifierMetric(config={})
        with pytest.raises(ValueError, match="llm_client"):
            metric.validate_config({
                "target_columns": ["query"],
                "prompts": [{"template": "x", "categories": ["a"], "target": "query"}],
            })

    def test_validate_config_valid(self):
        """Validate passes with proper config."""
        config = {
            "target_columns": ["query"],
            "prompts": [{"template": "x {text}", "categories": ["a"], "target": "query"}],
            "client_instance": MockLLMClient({}),
        }
        metric = LLMClassifierMetric(config=config)
        metric.validate_config(config)

    def test_process_query_classification(self, sample_df, context):
        """Test that LLM query classification produces correct boolean columns."""
        mock_responses = {
            0: "asking_a_question",
            1: "other",
            2: "asking_a_question",
        }
        mock_client = MockLLMClient(mock_responses)

        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "template": "Classify this message: {text}",
                    "categories": ["asking_a_question", "other"],
                    "target": "query",
                    "prefix": "llm_query",
                }
            ],
            "client_instance": mock_client,
        }
        metric = LLMClassifierMetric(config=config)
        result = metric.process(sample_df.copy(), context)

        # Check boolean columns were created
        assert "llm_query_asking_a_question" in result.columns
        assert "llm_query_other" in result.columns
        assert "llm_query_raw_response" in result.columns

        # Row 0: asking_a_question
        assert result.iloc[0]["llm_query_asking_a_question"] == True
        assert result.iloc[0]["llm_query_other"] == False

        # Row 1: other
        assert result.iloc[1]["llm_query_asking_a_question"] == False
        assert result.iloc[1]["llm_query_other"] == True

        # Row 2: asking_a_question
        assert result.iloc[2]["llm_query_asking_a_question"] == True
        assert result.iloc[2]["llm_query_other"] == False

    def test_process_response_classification(self, sample_df, context):
        """Test that LLM response classification works with multiple labels."""
        mock_responses = {
            0: "other",
            1: "not_in_knowledgebase",
            2: "clarification_needed",
        }
        mock_client = MockLLMClient(mock_responses)

        config = {
            "target_columns": ["response"],
            "prompts": [
                {
                    "template": "Classify this response: {text}",
                    "categories": ["not_in_knowledgebase", "clarification_needed", "other"],
                    "target": "response",
                    "prefix": "llm_resp",
                }
            ],
            "client_instance": mock_client,
        }
        metric = LLMClassifierMetric(config=config)
        result = metric.process(sample_df.copy(), context)

        # Row 1: not_in_knowledgebase
        assert result.iloc[1]["llm_resp_not_in_knowledgebase"] == True
        assert result.iloc[1]["llm_resp_clarification_needed"] == False

        # Row 2: clarification_needed
        assert result.iloc[2]["llm_resp_clarification_needed"] == True
        assert result.iloc[2]["llm_resp_not_in_knowledgebase"] == False

    def test_process_multi_label_response(self, sample_df, context):
        """Test parsing when LLM returns multiple comma-separated labels."""
        mock_responses = {
            0: "other",
            1: "not_in_knowledgebase, clarification_needed",
            2: "clarification_needed",
        }
        mock_client = MockLLMClient(mock_responses)

        config = {
            "target_columns": ["response"],
            "prompts": [
                {
                    "template": "Classify: {text}",
                    "categories": ["not_in_knowledgebase", "clarification_needed", "other"],
                    "target": "response",
                    "prefix": "llm_r",
                }
            ],
            "client_instance": mock_client,
        }
        metric = LLMClassifierMetric(config=config)
        result = metric.process(sample_df.copy(), context)

        # Row 1: both labels should be True
        assert result.iloc[1]["llm_r_not_in_knowledgebase"] == True
        assert result.iloc[1]["llm_r_clarification_needed"] == True
        assert result.iloc[1]["llm_r_other"] == False

    def test_process_invalid_response_ignored(self, sample_df, context):
        """Test that invalid/unexpected labels from LLM are filtered out."""
        mock_responses = {
            0: "asking_a_question",
            1: "INVALID_CATEGORY",
            2: "asking a question",  # has space instead of underscore
        }
        mock_client = MockLLMClient(mock_responses)

        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "template": "Classify: {text}",
                    "categories": ["asking_a_question", "other"],
                    "target": "query",
                    "prefix": "llm_q",
                }
            ],
            "client_instance": mock_client,
        }
        metric = LLMClassifierMetric(config=config)
        result = metric.process(sample_df.copy(), context)

        # Row 0: valid
        assert result.iloc[0]["llm_q_asking_a_question"] == True

        # Row 1: invalid category, both should be False
        assert result.iloc[1]["llm_q_asking_a_question"] == False
        assert result.iloc[1]["llm_q_other"] == False

        # Row 2: "asking a question" -> normalized to "asking_a_question" (spaces become underscores)
        assert result.iloc[2]["llm_q_asking_a_question"] == True

    def test_process_empty_response(self, sample_df, context):
        """Test handling of empty LLM responses."""
        mock_responses = {
            0: "",
            1: "",
            2: "",
        }
        mock_client = MockLLMClient(mock_responses)

        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "template": "Classify: {text}",
                    "categories": ["asking_a_question", "other"],
                    "target": "query",
                    "prefix": "llm_q",
                }
            ],
            "client_instance": mock_client,
        }
        metric = LLMClassifierMetric(config=config)
        result = metric.process(sample_df.copy(), context)

        # All should be False when response is empty
        assert all(result["llm_q_asking_a_question"] == False)
        assert all(result["llm_q_other"] == False)

    def test_safer_llm_prompts_yaml_loads(self):
        """Test that the SAFER LLM prompts YAML loads correctly."""
        import yaml
        from post_deploy.presets.safer import PRESET_DIR

        prompts_path = PRESET_DIR / "llm_prompts.yaml"
        assert prompts_path.exists()

        with open(prompts_path) as f:
            prompts = yaml.safe_load(f)

        assert isinstance(prompts, list)
        assert len(prompts) == 2

        # Query prompt
        assert prompts[0]["target"] == "query"
        assert "asking_a_question" in prompts[0]["categories"]
        assert "other" in prompts[0]["categories"]
        assert "{text}" in prompts[0]["template"]

        # Response prompt
        assert prompts[1]["target"] == "response"
        assert "not_in_knowledgebase" in prompts[1]["categories"]
        assert "clarification_needed" in prompts[1]["categories"]
        assert "{text}" in prompts[1]["template"]


class TestParseResponses:
    """Tests for the response parsing logic."""

    def test_single_valid_label(self):
        """Test parsing a single valid label."""
        responses = {0: "asking_a_question"}
        categories = ["asking_a_question", "other"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert result[0] == ["asking_a_question"]

    def test_multiple_valid_labels(self):
        """Test parsing multiple comma-separated valid labels."""
        responses = {0: "not_in_knowledgebase, clarification_needed"}
        categories = ["not_in_knowledgebase", "clarification_needed", "other"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert set(result[0]) == {"not_in_knowledgebase", "clarification_needed"}

    def test_invalid_labels_filtered(self):
        """Test that invalid labels are filtered out."""
        responses = {0: "valid_label, GARBAGE, another_invalid"}
        categories = ["valid_label", "other"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert result[0] == ["valid_label"]

    def test_empty_response(self):
        """Test that empty response produces empty list."""
        responses = {0: ""}
        categories = ["a", "b"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert result[0] == []

    def test_whitespace_normalization(self):
        """Test that whitespace around labels is stripped."""
        responses = {0: "  asking_a_question  ,  other  "}
        categories = ["asking_a_question", "other"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert set(result[0]) == {"asking_a_question", "other"}

    def test_space_to_underscore_normalization(self):
        """Test that spaces in labels are normalized to underscores."""
        responses = {0: "asking a question"}
        categories = ["asking_a_question", "other"]
        result = LLMClassifierMetric._parse_responses(responses, categories)
        assert result[0] == ["asking_a_question"]
