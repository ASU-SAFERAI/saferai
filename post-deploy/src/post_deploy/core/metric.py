"""Base metric interface that all metrics must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd


class MetricContext:
    """
    Runtime context passed to metrics during processing.

    Holds column mappings and shared pipeline state so metrics
    can resolve target columns dynamically rather than hardcoding names.
    """

    def __init__(
        self,
        columns: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """
        Args:
            columns: Mapping of logical column names to actual DataFrame column names.
                     Example: {"query": "query_raw", "response": "response_raw"}
            extra: Arbitrary additional context for metric-specific needs.
        """
        self.columns = columns or {}
        self.extra = extra or {}

    def get_column(self, logical_name: str) -> str:
        """
        Resolve a logical column name to the actual DataFrame column name.

        Raises:
            KeyError: If the logical name is not mapped.
        """
        if logical_name not in self.columns:
            raise KeyError(
                f"Column '{logical_name}' not found in context. "
                f"Available mappings: {list(self.columns.keys())}"
            )
        return self.columns[logical_name]


class BaseMetric(ABC):
    """
    Abstract base class for all metrics.

    A metric takes a DataFrame and returns it with new columns appended
    representing the metric's findings. Metrics are stateless processors:
    configuration is provided at init, data flows through `process()`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this metric (e.g., 'keyword_search')."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of this metric implementation."""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> list[str]:
        """
        Logical column names this metric requires from the context.

        These are logical names (e.g., 'query', 'response') that will be
        resolved to actual DataFrame column names via MetricContext.
        """
        ...

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """
        Validate metric-specific configuration.

        Called before processing begins. Should raise ValueError
        with a descriptive message if config is invalid.

        Args:
            config: Metric-specific configuration dictionary.
        """
        ...

    @abstractmethod
    def process(self, df: pd.DataFrame, context: MetricContext) -> pd.DataFrame:
        """
        Execute this metric on the input DataFrame.

        Must return a new DataFrame (or the same one mutated) with
        additional columns representing this metric's output. Should
        not drop or modify existing columns.

        Args:
            df: Input DataFrame to process.
            context: Runtime context with column mappings and extras.

        Returns:
            DataFrame with metric result columns appended.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} version={self.version!r}>"
