"""Post-processing utilities: transform wide-format metric results into various output formats."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from .config import OutputFormat, PostProcessConfig

logger = logging.getLogger(__name__)


def post_process_dataframe(
    df: pd.DataFrame,
    config: PostProcessConfig,
    output_format: OutputFormat = OutputFormat.LONG,
    engine_version: str | None = None,
) -> pd.DataFrame:
    """
    Apply post-processing to a wide-format metric DataFrame.

    Depending on output_format:
    - WIDE: returns the DataFrame as-is (no transformation)
    - LONG: melts all non-ID columns into metric_name/metric_value rows

    Args:
        df: Wide-format DataFrame with metric columns.
        config: Post-processing configuration.
        output_format: Target output format.
        engine_version: Version string to stamp on output rows.

    Returns:
        Processed DataFrame in the requested format.
    """
    if not config.enabled:
        return df

    if output_format == OutputFormat.WIDE:
        return _apply_wide_format(df, config, engine_version)

    return _melt_to_long(df, config, engine_version)


def _apply_wide_format(
    df: pd.DataFrame,
    config: PostProcessConfig,
    engine_version: str | None = None,
) -> pd.DataFrame:
    """
    Apply wide-format post-processing: add metadata columns without melting.

    Args:
        df: Input DataFrame.
        config: Post-processing configuration.
        engine_version: Version string.

    Returns:
        DataFrame with metadata columns appended.
    """
    from post_deploy import __version__

    version = engine_version or __version__
    df = df.copy()
    df[config.version_column] = version
    df[config.run_day_column] = date.today().strftime("%Y-%m-%d")

    return df


def _melt_to_long(
    df: pd.DataFrame,
    config: PostProcessConfig,
    engine_version: str | None = None,
) -> pd.DataFrame:
    """
    Melt a wide DataFrame into long format with metric_name/metric_value columns.

    Steps:
    1. Filter id_cols to only those present in df
    2. Drop configured drop_cols
    3. Melt remaining columns into metric_name/metric_value
    4. Add version and run_day metadata

    Args:
        df: Wide-format DataFrame.
        config: Post-processing configuration.
        engine_version: Version string to stamp on output.

    Returns:
        Long-format DataFrame.

    Raises:
        ValueError: If no metric columns remain after dropping.
    """
    from post_deploy import __version__

    version = engine_version or __version__

    # Only keep id_cols that actually exist in the DataFrame
    id_cols = [col for col in config.id_cols if col in df.columns]

    # Drop specified columns (e.g., raw text columns)
    cols_to_drop = [col for col in config.drop_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.debug("Dropped columns: %s", cols_to_drop)

    # Determine metric columns (everything not in id_cols)
    metric_cols = [col for col in df.columns if col not in id_cols]

    if not metric_cols:
        raise ValueError(
            "No metric columns remain after dropping specified columns. "
            "Check post_process.id_cols and post_process.drop_cols configuration."
        )

    logger.info(
        "Melting to long format: %d id columns, %d metric columns.",
        len(id_cols), len(metric_cols),
    )

    df_long = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="metric_value",
    )

    # Add metadata
    df_long[config.version_column] = version
    df_long[config.run_day_column] = date.today().strftime("%Y-%m-%d")

    return df_long
