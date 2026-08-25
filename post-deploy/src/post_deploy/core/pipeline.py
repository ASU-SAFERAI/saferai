"""Pipeline orchestrator: loads metrics, runs them, handles post-processing and I/O."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from .config import PipelineConfig
from .metric import BaseMetric, MetricContext
from .post_process import post_process_dataframe
from .registry import MetricRegistry, default_registry

if TYPE_CHECKING:
    from post_deploy.io.base import InputSource, OutputManager

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates the full post-deploy processing pipeline.

    1. Reads DataFrames from an InputSource
    2. Runs each enabled metric in sequence
    3. Optionally post-processes results (wide → long melt)
    4. Writes output via an OutputManager
    """

    def __init__(
        self,
        config: PipelineConfig,
        input_source: "InputSource",
        output_manager: "OutputManager",
        registry: MetricRegistry | None = None,
    ):
        """
        Args:
            config: Validated pipeline configuration.
            input_source: Source that yields (name, DataFrame) tuples.
            output_manager: Destination for processed DataFrames.
            registry: Metric registry to use. Defaults to the global registry.
        """
        self.config = config
        self.input_source = input_source
        self.output_manager = output_manager
        self.registry = registry or default_registry

        # Build metric instances
        self._metrics = self._load_metrics()

        # Build the context from column mappings
        self._context = MetricContext(
            columns=config.input.columns,
            extra={"pipeline_version": config.version},
        )

    def _load_metrics(self) -> list[BaseMetric]:
        """Instantiate and validate all enabled metrics."""
        metrics: list[BaseMetric] = []

        for metric_config in self.config.get_enabled_metrics():
            metric_cls = self.registry.get(metric_config.name)
            instance = metric_cls(config=metric_config.config)
            instance.validate_config(metric_config.config)
            metrics.append(instance)
            logger.info("Loaded metric: %s (v%s)", instance.name, instance.version)

        if not metrics:
            logger.warning("No metrics enabled in pipeline configuration.")

        return metrics

    def run(self) -> None:
        """
        Execute the full pipeline: read → process → post-process → write.

        Iterates over all DataFrames from the input source, applies each
        metric sequentially, then writes the result.
        """
        logger.info("Starting pipeline with %d metric(s).", len(self._metrics))

        target_cols = list(self.config.input.columns.values())

        for name, df in self.input_source.get_cleaned_dataframes(target_cols=target_cols):
            logger.info("Processing DataFrame '%s' (shape: %s)", name, df.shape)

            df = self._run_metrics(df)

            if self.config.post_process.enabled:
                df = self._post_process(df)

            self.output_manager.save(name, df)
            logger.info("Saved output for '%s'.", name)

        logger.info("Pipeline complete.")

    def run_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the pipeline on a single DataFrame without I/O.

        Useful for programmatic usage where you already have data in memory.

        Args:
            df: Input DataFrame.

        Returns:
            Processed DataFrame (post-processed if configured).
        """
        df = self._run_metrics(df)

        if self.config.post_process.enabled:
            df = self._post_process(df)

        return df

    def _run_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled metrics sequentially to the DataFrame."""
        for metric in self._metrics:
            logger.info("Running metric: %s", metric.name)

            # Validate required columns exist
            self._check_required_columns(df, metric)

            df = metric.process(df, self._context)
            logger.debug("DataFrame shape after '%s': %s", metric.name, df.shape)

        return df

    def _check_required_columns(self, df: pd.DataFrame, metric: BaseMetric) -> None:
        """Verify that all columns required by a metric are present."""
        for logical_col in metric.required_columns:
            actual_col = self._context.get_column(logical_col)
            if actual_col not in df.columns:
                raise ValueError(
                    f"Metric '{metric.name}' requires column '{logical_col}' "
                    f"(mapped to '{actual_col}'), but it's not in the DataFrame. "
                    f"Available columns: {list(df.columns)}"
                )

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to the metric results.

        Delegates to the standalone post_process_dataframe function,
        which respects the output format and post-process config.
        """
        return post_process_dataframe(
            df=df,
            config=self.config.post_process,
            output_format=self.config.output.format,
        )
