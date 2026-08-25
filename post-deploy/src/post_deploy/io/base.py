"""Abstract base classes for input sources and output managers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class InputSource(ABC):
    """
    Base class for input sources.

    Input sources provide DataFrames to the pipeline. They handle
    data loading and basic cleaning (null/empty row removal).
    """

    @abstractmethod
    def load_raw_dataframes(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """
        Generator that yields tuples of (name, DataFrame) from the input source.

        DataFrames are loaded without further cleaning.

        Yields:
            Tuple of (identifier_name, raw_dataframe).
        """
        ...

    def get_cleaned_dataframes(self, target_cols: list[str]) -> Iterable[Tuple[str, pd.DataFrame]]:
        """
        Generator that yields cleaned DataFrames.

        Drops rows where any target column is null or empty string.

        Args:
            target_cols: List of column names that must be non-null and non-empty.

        Yields:
            Tuple of (identifier_name, cleaned_dataframe).

        Raises:
            ValueError: If a DataFrame is empty after cleaning or missing target columns.
        """
        for name, df in self.load_raw_dataframes():
            cleaned_df = self._clean_dataframe(df, target_cols)

            if cleaned_df.empty:
                raise ValueError(f"DataFrame '{name}' is empty after cleaning.")

            yield name, cleaned_df

    def _clean_dataframe(self, df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
        """
        Drop rows where target columns are null or empty strings.

        Args:
            df: Raw DataFrame.
            target_cols: Columns that must have values.

        Returns:
            Cleaned DataFrame.
        """
        missing_cols = [col for col in target_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing target columns in DataFrame: {missing_cols}")

        row_count_original = len(df)
        logger.info("DataFrame shape before cleaning: %s", df.shape)

        df = df.dropna(subset=target_cols)
        df = df[
            df[target_cols].apply(lambda col: col.astype(str).str.strip() != "").all(axis=1)
        ]

        row_count_new = len(df)
        logger.info(
            "Dropped %d rows with null or empty target columns.", row_count_original - row_count_new
        )
        logger.info("DataFrame shape after cleaning: %s", df.shape)

        return df


class OutputManager(ABC):
    """
    Base class for output managers.

    Output managers handle writing processed DataFrames to their destination.
    """

    @abstractmethod
    def save(self, name: str, df: pd.DataFrame) -> None:
        """
        Save a processed DataFrame.

        Args:
            name: Identifier for the DataFrame (e.g., source filename).
            df: Processed DataFrame to save.
        """
        ...
