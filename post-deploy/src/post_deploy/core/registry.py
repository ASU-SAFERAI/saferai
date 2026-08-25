"""Metric registry: discover, register, and load metric implementations."""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import Type

from .metric import BaseMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """
    Central registry for metric classes.

    Supports two modes of registration:
    1. Explicit: call `registry.register(MyMetric)` directly.
    2. Entry points: auto-discover metrics installed via the
       `post_deploy.metrics` entry point group.
    """

    def __init__(self):
        self._metrics: dict[str, Type[BaseMetric]] = {}
        self._entry_points_loaded = False

    def register(self, metric_cls: Type[BaseMetric]) -> None:
        """
        Register a metric class by its `name` property.

        Args:
            metric_cls: A class that implements BaseMetric.

        Raises:
            TypeError: If metric_cls doesn't subclass BaseMetric.
            ValueError: If a metric with the same name is already registered.
        """
        if not (isinstance(metric_cls, type) and issubclass(metric_cls, BaseMetric)):
            raise TypeError(
                f"Expected a subclass of BaseMetric, got {metric_cls!r}"
            )

        # Instantiate temporarily to read the name property
        # We store the class, not the instance
        name = self._get_metric_name(metric_cls)

        if name in self._metrics:
            existing = self._metrics[name]
            if existing is metric_cls:
                return  # Already registered, idempotent
            raise ValueError(
                f"Metric '{name}' is already registered by {existing!r}. "
                f"Cannot register {metric_cls!r} with the same name."
            )

        self._metrics[name] = metric_cls
        logger.debug("Registered metric: %s -> %s", name, metric_cls)

    def get(self, name: str) -> Type[BaseMetric]:
        """
        Retrieve a metric class by name.

        Loads entry points on first access if not already loaded.

        Args:
            name: The metric name (e.g., 'keyword_search').

        Returns:
            The metric class.

        Raises:
            KeyError: If no metric with that name is registered.
        """
        if name not in self._metrics and not self._entry_points_loaded:
            self._load_entry_points()

        if name not in self._metrics:
            available = ", ".join(sorted(self._metrics.keys())) or "(none)"
            raise KeyError(
                f"Metric '{name}' not found. Available metrics: {available}"
            )

        return self._metrics[name]

    def list_metrics(self) -> list[str]:
        """Return a sorted list of all registered metric names."""
        if not self._entry_points_loaded:
            self._load_entry_points()
        return sorted(self._metrics.keys())

    def load_all(self, metric_names: list[str], configs: dict[str, dict] | None = None) -> list[BaseMetric]:
        """
        Load and instantiate multiple metrics by name.

        Args:
            metric_names: List of metric names to load.
            configs: Optional dict mapping metric name -> config dict.

        Returns:
            List of instantiated BaseMetric objects.
        """
        configs = configs or {}
        instances = []

        for name in metric_names:
            metric_cls = self.get(name)
            config = configs.get(name, {})
            instance = metric_cls(config=config)
            instances.append(instance)

        return instances

    def _load_entry_points(self) -> None:
        """Discover and load metrics from the 'post_deploy.metrics' entry point group."""
        self._entry_points_loaded = True

        try:
            eps = entry_points(group="post_deploy.metrics")
        except TypeError:
            # Python 3.9 compat: entry_points() doesn't accept group kwarg
            all_eps = entry_points()
            eps = all_eps.get("post_deploy.metrics", [])

        for ep in eps:
            try:
                metric_cls = ep.load()
                if isinstance(metric_cls, type) and issubclass(metric_cls, BaseMetric):
                    name = self._get_metric_name(metric_cls)
                    # Don't overwrite explicitly registered metrics
                    if name not in self._metrics:
                        self._metrics[name] = metric_cls
                        logger.debug("Loaded metric from entry point: %s -> %s", name, metric_cls)
                else:
                    logger.warning(
                        "Entry point '%s' does not point to a BaseMetric subclass, skipping.",
                        ep.name,
                    )
            except Exception as e:
                logger.warning("Failed to load metric entry point '%s': %s", ep.name, e)

    @staticmethod
    def _get_metric_name(metric_cls: Type[BaseMetric]) -> str:
        """
        Extract the metric name from a class.

        Tries the class attribute first; falls back to a temporary instantiation.
        """
        # Check if name is a class-level attribute (not just an abstract property)
        if hasattr(metric_cls, "name") and isinstance(
            getattr(metric_cls, "name", None), str
        ):
            return metric_cls.name  # type: ignore[return-value]

        # For classes where name is defined as a property or in __init__,
        # we use a convention: class attribute NAME or fall back to class name
        if hasattr(metric_cls, "NAME"):
            return metric_cls.NAME  # type: ignore[return-value]

        # Last resort: derive from class name
        # e.g., KeywordSearchMetric -> keyword_search
        cls_name = metric_cls.__name__
        if cls_name.endswith("Metric"):
            cls_name = cls_name[:-6]

        # Convert CamelCase to snake_case
        import re
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()
        return name


# Global registry instance
default_registry = MetricRegistry()
