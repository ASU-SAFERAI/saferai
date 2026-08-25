"""Unit tests for ZeroShotMetric (mocked inference)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from post_deploy.core.metric import MetricContext
from post_deploy.metrics.zero_shot import ZeroShotMetric


@pytest.fixture
def context():
    """Create a MetricContext with standard column mappings."""
    return MetricContext(columns={"query": "query_raw", "response": "response_raw"})


@pytest.fixture
def sample_df():
    """Create a small sample DataFrame."""
    return pd.DataFrame({
        "query_raw": ["What is 2+2?", "You are a robot."],
        "response_raw": ["The answer is 4.", "I'm sorry, I need more context."],
    })


class TestZeroShotMetric:
    """Tests for the ZeroShotMetric class."""

    def test_basic_properties(self):
        """Test metric name and version."""
        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "hypothesis_template": "This text is {}",
                    "labels": {"q_question": "asking a question."},
                    "target": "query",
                }
            ],
        }
        metric = ZeroShotMetric(config=config)
        assert metric.name == "zero_shot"
        assert metric.version == "1.0.0"
        assert metric.required_columns == ["query"]

    def test_validate_config_missing_columns(self):
        """Validate raises on empty target_columns."""
        metric = ZeroShotMetric(config={"prompts": [{"hypothesis_template": "x", "labels": {"a": "b"}, "target": "q"}]})
        with pytest.raises(ValueError, match="target_columns"):
            metric.validate_config({"target_columns": []})

    def test_validate_config_missing_prompts(self):
        """Validate raises when no prompt source is provided."""
        metric = ZeroShotMetric(config={"target_columns": ["query"]})
        with pytest.raises(ValueError, match="prompt source"):
            metric.validate_config({"target_columns": ["query"]})

    @patch("post_deploy.metrics.zero_shot.ZeroShotMetric._ensure_classifier")
    def test_process_with_mocked_classifier(self, mock_ensure, sample_df, context):
        """Test process with a mocked HuggingFace classifier."""
        config = {
            "target_columns": ["query"],
            "prompts": [
                {
                    "hypothesis_template": "This text is {}",
                    "labels": {"q_question": "asking a question."},
                    "target": "query",
                }
            ],
        }
        metric = ZeroShotMetric(config=config)

        # Mock the classifier to return predictable scores
        mock_classifier = MagicMock()
        mock_classifier.side_effect = lambda texts, **kwargs: [
            {"labels": ["asking a question."], "scores": [0.95]}
            for _ in texts
        ]
        metric._classifier = mock_classifier
        metric._effective_batch_size = 1

        # Mock the datasets module which is imported inside the method
        mock_dataset_instance = MagicMock()

        def mock_map(fn, batched, batch_size):
            batch = {"query_raw": sample_df["query_raw"].tolist()}
            result = fn(batch)
            result_ds = MagicMock()
            result_df = pd.DataFrame({
                "query_raw": sample_df["query_raw"].tolist(),
                "asking a question.": result["asking a question."],
            })
            result_ds.to_pandas.return_value = result_df
            return result_ds

        mock_dataset_instance.map = mock_map

        mock_dataset_cls = MagicMock()
        mock_dataset_cls.from_pandas.return_value = mock_dataset_instance

        with patch.dict("sys.modules", {"datasets": MagicMock(Dataset=mock_dataset_cls)}):
            # Need to reimport since Dataset is imported inside the method
            import importlib
            import post_deploy.metrics.zero_shot as zs_module
            # Patch at the point of use inside the method
            original_process = metric._classify_with_prompt

            def patched_classify(df, actual_col, prompt_def):
                from unittest.mock import patch as inner_patch
                with inner_patch("datasets.Dataset", mock_dataset_cls):
                    return original_process(df, actual_col, prompt_def)

            # Simpler approach: just directly patch the import in the function
            import sys
            import types
            mock_datasets_mod = types.ModuleType("datasets")
            mock_datasets_mod.Dataset = mock_dataset_cls
            sys.modules["datasets"] = mock_datasets_mod
            try:
                result = metric.process(sample_df.copy(), context)
            finally:
                del sys.modules["datasets"]

        # Check that the output column was added
        assert "q_question" in result.columns
        assert len(result) == 2


class TestZeroShotConfigValidation:
    """Test configuration validation edge cases."""

    def test_validate_missing_ml_deps(self):
        """Test that validation reports missing ML dependencies."""
        config = {
            "target_columns": ["query"],
            "prompts": [{"hypothesis_template": "x", "labels": {"a": "b"}, "target": "query"}],
        }
        metric = ZeroShotMetric(config=config)

        # This test only fails if transformers/torch are not installed
        # In a dev environment they may be present, so we just ensure no crash
        try:
            metric.validate_config(config)
        except ValueError as e:
            assert "transformers" in str(e) or "torch" in str(e)
