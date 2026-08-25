"""PII search metric: detect personally identifiable information using Presidio."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from post_deploy.core.metric import BaseMetric, MetricContext

logger = logging.getLogger(__name__)

# Default entity types to search for
DEFAULT_ENTITY_TYPES = ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER", "URL"]


class PiiSearchMetric(BaseMetric):
    """
    Detects Personally Identifiable Information (PII) in text columns using Microsoft Presidio.

    Requires the optional `presidio_analyzer` dependency:
        pip install post-deploy[pii]

    Configuration options:
        target_columns: list[str] - logical column names to scan (e.g., ["query"])
        entity_types: list[str] - Presidio entity types to detect
            Default: ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER", "URL"]
            Full list: https://microsoft.github.io/presidio/supported_entities/
        language: str - language code for analysis (default: "en")
        verbose: bool - include per-entity-type detail columns (default: True)
    """

    NAME = "pii_search"

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._target_columns: list[str] = self._config.get("target_columns", ["query"])
        self._entity_types: list[str] = self._config.get("entity_types", DEFAULT_ENTITY_TYPES)
        self._language: str = self._config.get("language", "en")
        self._verbose: bool = self._config.get("verbose", True)
        self._analyzer = None  # Lazy-loaded

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
        """Validate PII search configuration."""
        target_columns = config.get("target_columns", ["query"])
        if not target_columns:
            raise ValueError("PiiSearchMetric requires at least one 'target_columns' entry.")

        entity_types = config.get("entity_types", DEFAULT_ENTITY_TYPES)
        if not entity_types:
            raise ValueError("PiiSearchMetric requires at least one entity type in 'entity_types'.")

        # Validate that presidio is available
        try:
            import presidio_analyzer  # noqa: F401
        except ImportError:
            raise ValueError(
                "PiiSearchMetric requires 'presidio_analyzer'. "
                "Install it with: pip install post-deploy[pii]"
            )

    def process(self, df: pd.DataFrame, context: MetricContext) -> pd.DataFrame:
        """Run PII detection on all configured target columns."""
        self._ensure_analyzer()

        for logical_col in self._target_columns:
            actual_col = context.get_column(logical_col)
            df = self._search_column(df, actual_col, logical_col)

        return df

    def _search_column(self, df: pd.DataFrame, actual_col: str, logical_col: str) -> pd.DataFrame:
        """Run PII detection on a single column."""
        logger.info("Running PII search on column '%s' (entities: %s)", actual_col, self._entity_types)

        prefix = f"pii_{logical_col}"

        if df.empty:
            # Add expected columns with correct dtypes for empty DataFrames
            df[f"{prefix}_found_all"] = pd.Series(dtype=object)
            df[f"{prefix}_found_distinct"] = pd.Series(dtype=object)
            df[f"{prefix}_entities_found"] = pd.Series(dtype=object)
            df[f"{prefix}_any_found"] = pd.Series(dtype=bool)
            return df

        text_series = df[actual_col].astype(str)

        results = text_series.apply(self._analyze_text)

        # Expand result dicts into columns
        result_df = pd.DataFrame(results.tolist(), index=df.index)

        # Rename columns with prefix
        rename_map = {col: f"{prefix}_{col}" for col in result_df.columns}
        result_df = result_df.rename(columns=rename_map)

        df = pd.concat([df, result_df], axis=1)
        return df

    def _analyze_text(self, text: str) -> dict[str, Any]:
        """
        Analyze a single text string for PII entities.

        Returns a dict with:
            - found_all: list of all PII text found
            - found_distinct: deduplicated list
            - entities_found: entity types detected
            - any_found: bool
            - (if verbose) ne_{entity_type}: list of found text per entity type
        """
        results: dict[str, Any] = {
            "found_all": [],
            "found_distinct": [],
            "entities_found": [],
            "any_found": False,
        }

        analyzer_results = self._analyzer.analyze(
            text=text,
            entities=self._entity_types,
            language=self._language,
        )

        if not analyzer_results:
            return results

        found_list: list[str] = []
        entities_found: list[str] = []
        entity_details: dict[str, list[str]] = {}

        for res in analyzer_results:
            if res.entity_type in self._entity_types:
                identified_text = text[res.start:res.end].lower()
                found_list.append(identified_text)
                entities_found.append(res.entity_type)

                if self._verbose:
                    key = f"ne_{res.entity_type.lower()}"
                    if key not in entity_details:
                        entity_details[key] = []
                    entity_details[key].append(text[res.start:res.end])

        results["found_all"] = found_list
        results["found_distinct"] = list(set(found_list))
        results["entities_found"] = list(set(entities_found))
        results["any_found"] = len(found_list) > 0

        if self._verbose:
            results.update(entity_details)

        return results

    def _ensure_analyzer(self) -> None:
        """Lazy-load the Presidio AnalyzerEngine."""
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
            except ImportError:
                raise RuntimeError(
                    "presidio_analyzer is not installed. "
                    "Install it with: pip install post-deploy[pii]"
                )
            self._analyzer = AnalyzerEngine()
            logger.info("Initialized Presidio AnalyzerEngine.")
