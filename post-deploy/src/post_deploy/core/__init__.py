"""Core framework components: metric base class, registry, pipeline, config."""

from .metric import BaseMetric
from .registry import MetricRegistry
from .pipeline import Pipeline

__all__ = ["BaseMetric", "MetricRegistry", "Pipeline"]
