"""Local filesystem I/O: read CSVs from disk, write results to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

from .base import InputSource, OutputManager

logger = logging.getLogger(__name__)


class LocalCSVInputSource(InputSource):
    """
    Input source for reading local CSV files.

    Args:
        file_paths: Comma-separated file paths or a list of paths.
        encoding: File encoding (default: UTF-8).
    """

    def __init__(self, file_paths: str | list[str], encoding: str = "UTF-8"):
        if isinstance(file_paths, str):
            self.paths = [p.strip() for p in file_paths.split(",")]
        else:
            self.paths = file_paths
        self.encoding = encoding

    def load_raw_dataframes(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """Load each CSV file and yield (filename_stem, DataFrame)."""
        logger.info("Loading %d local CSV file(s).", len(self.paths))

        for path_str in self.paths:
            path = Path(path_str)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            name = path.stem
            logger.info("Reading file: %s", path)
            df = pd.read_csv(path, encoding=self.encoding)
            yield name, df


class LocalOutputManager(OutputManager):
    """
    Output manager that writes processed DataFrames to local CSV files.

    Args:
        output_dir: Directory to write output files.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)

    def save(self, name: str, df: pd.DataFrame) -> None:
        """Write DataFrame to a CSV file in the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"processed_{name}.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved DataFrame '%s' to: %s", name, output_path)
