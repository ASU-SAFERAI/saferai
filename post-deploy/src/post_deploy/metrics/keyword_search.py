"""Keyword search metric: regex-based keyword matching against text columns."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from post_deploy.core.metric import BaseMetric, MetricContext

logger = logging.getLogger(__name__)


class KeywordSearchMetric(BaseMetric):
    """
    Searches text columns for lists of keywords using word-boundary regex matching.

    Configuration accepts keyword groups either inline or from a YAML file.
    Each group produces two output columns:
      - {column}_{group_name}_found (bool): whether any keyword matched
      - {column}_{group_name}_matches (list[str]): keywords that matched

    Config options:
        target_columns: list[str] - logical column names to search (e.g., ["query", "response"])
        keyword_groups: dict[str, list[str]] - inline keyword groups
            Example: {"confusion": ["confused", "rephrase"], "error": ["error", "issue"]}
        keywords_file: str - path to a YAML file with keyword groups
        preset: str - name of a preset to load keywords from (e.g., "safer")
    """

    NAME = "keyword_search"

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._keyword_groups: dict[str, list[str]] = {}
        self._target_columns: list[str] = []
        self._load_config()

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required_columns(self) -> list[str]:
        return self._target_columns

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate that at least one keyword source and target column is configured."""
        target_columns = config.get("target_columns", [])
        has_keywords = (
            config.get("keyword_groups")
            or config.get("keywords_file")
            or config.get("preset")
        )

        if not target_columns:
            raise ValueError(
                "KeywordSearchMetric requires 'target_columns' in config. "
                "Example: ['query', 'response']"
            )

        if not has_keywords:
            raise ValueError(
                "KeywordSearchMetric requires at least one keyword source: "
                "'keyword_groups' (inline dict), 'keywords_file' (YAML path), "
                "or 'preset' (preset name)."
            )

    def process(self, df: pd.DataFrame, context: MetricContext) -> pd.DataFrame:
        """Run keyword search on all configured target columns."""
        for logical_col in self._target_columns:
            actual_col = context.get_column(logical_col)
            df = self._search_column(df, actual_col, logical_col)

        return df

    def _search_column(self, df: pd.DataFrame, actual_col: str, logical_col: str) -> pd.DataFrame:
        """Search a single column for all keyword groups."""
        logger.info("Running keyword search on column '%s' (%s groups)", actual_col, len(self._keyword_groups))

        # Ensure column is string type
        text_series = df[actual_col].astype(str)

        for group_name, keywords in self._keyword_groups.items():
            found_col = f"{logical_col}_{group_name}_found"
            matches_col = f"{logical_col}_{group_name}_matches"

            results = text_series.apply(lambda text: self._match_keywords(text, keywords))
            df[found_col] = results.apply(lambda r: r[0])
            df[matches_col] = results.apply(lambda r: r[1])

        return df

    @staticmethod
    def _match_keywords(text: str, keywords: list[str]) -> tuple[bool, list[str]]:
        """
        Check text for keyword matches using word-boundary regex.

        Args:
            text: Text to search.
            keywords: List of keywords/phrases to look for.

        Returns:
            Tuple of (any_found, list_of_matched_keywords).
        """
        lower_text = text.lower()
        matched = []

        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, lower_text):
                matched.append(kw.lower())

        return bool(matched), matched

    def _load_config(self) -> None:
        """Load keyword groups and target columns from config."""
        self._target_columns = self._config.get("target_columns", [])

        # Load keywords from multiple sources (they merge)
        self._keyword_groups = {}

        # 1. Inline keyword groups
        inline_groups = self._config.get("keyword_groups", {})
        if inline_groups:
            self._keyword_groups.update(inline_groups)

        # 2. From YAML file
        keywords_file = self._config.get("keywords_file")
        if keywords_file:
            file_groups = self._load_keywords_from_yaml(keywords_file)
            self._keyword_groups.update(file_groups)

        # 3. From preset
        preset_name = self._config.get("preset")
        if preset_name:
            preset_groups = self._load_keywords_from_preset(preset_name)
            self._keyword_groups.update(preset_groups)

    @staticmethod
    def _load_keywords_from_yaml(path: str) -> dict[str, list[str]]:
        """Load keyword groups from a YAML file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Keywords file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Keywords YAML must be a mapping of group_name -> keyword_list, got {type(data).__name__}")

        # Validate structure
        for group_name, keywords in data.items():
            if not isinstance(keywords, list):
                raise ValueError(f"Keyword group '{group_name}' must be a list, got {type(keywords).__name__}")

        return data

    @staticmethod
    def _load_keywords_from_preset(preset_name: str) -> dict[str, list[str]]:
        """Load keyword groups from a named preset."""
        from post_deploy.presets import safer as safer_preset

        if preset_name == "safer":
            keywords_path = safer_preset.PRESET_DIR / "keywords.yaml"
            if not keywords_path.exists():
                raise FileNotFoundError(
                    f"SAFER preset keywords file not found at {keywords_path}. "
                    "Run the preset setup or provide keywords inline."
                )
            with open(keywords_path) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown preset: '{preset_name}'. Available presets: ['safer']")
