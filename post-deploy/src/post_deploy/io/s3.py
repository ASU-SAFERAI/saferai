"""S3 I/O: read and write CSVs from/to Amazon S3."""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Iterable, Tuple

import pandas as pd

from .base import InputSource, OutputManager

logger = logging.getLogger(__name__)


class S3InputSource(InputSource):
    """
    Input source for reading CSV files from an S3 prefix.

    Requires: pip install post-deploy[aws]

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix to search for CSV files.
        encoding: File encoding (default: UTF-8).
    """

    def __init__(self, bucket: str, prefix: str = "", encoding: str = "UTF-8"):
        self.bucket = bucket
        self.prefix = prefix
        self.encoding = encoding

    def load_raw_dataframes(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """Discover and load CSV files from the S3 prefix."""
        try:
            import s3fs
        except ImportError:
            raise RuntimeError(
                "s3fs is required for S3 input. Install with: pip install post-deploy[aws]"
            )

        fs = s3fs.S3FileSystem()
        s3_prefix = f"{self.bucket}/{self.prefix}".rstrip("/")

        logger.info("Listing CSV files at s3://%s", s3_prefix)
        all_files = fs.ls(s3_prefix, detail=False)
        csv_files = [f for f in all_files if f.endswith(".csv")]

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found at s3://{s3_prefix}")

        logger.info("Found %d CSV file(s) in S3.", len(csv_files))

        for s3_path in csv_files:
            name = PurePosixPath(s3_path).stem
            full_uri = f"s3://{s3_path}"
            logger.info("Reading: %s", full_uri)
            df = pd.read_csv(full_uri, encoding=self.encoding)
            yield name, df


class S3OutputManager(OutputManager):
    """
    Output manager that writes processed DataFrames to S3.

    Requires: pip install post-deploy[aws]

    Args:
        bucket: S3 bucket name.
        prefix: S3 key prefix for output files.
    """

    def __init__(self, bucket: str, prefix: str = "output"):
        self.bucket = bucket
        self.prefix = prefix

    def save(self, name: str, df: pd.DataFrame) -> None:
        """Write DataFrame to a CSV file in S3."""
        s3_path = f"s3://{self.bucket}/{self.prefix}/processed_{name}.csv"
        df.to_csv(s3_path, index=False)
        logger.info("Saved DataFrame '%s' to S3: %s", name, s3_path)
