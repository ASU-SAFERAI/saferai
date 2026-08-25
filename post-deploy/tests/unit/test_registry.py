"""Unit tests for MetricRegistry."""

from __future__ import annotations

import pytest

from post_deploy.core.metric import BaseMetric, MetricContext
from post_deploy.core.registry import MetricRegistry


class DummyMetric(BaseMetric):
    """A minimal metric implementation for testing."""

    NAME = "dummy"

    def __init__(self, config=None):
        self._config = config or {}

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_columns(self) -> list[str]:
        return ["query"]

    def validate_config(self, config):
        pass

    def process(self, df, context):
        df["dummy_output"] = True
        return df


class AnotherMetric(BaseMetric):
    """Another metric for testing duplicate registration."""

    NAME = "another"

    def __init__(self, config=None):
        pass

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def required_columns(self) -> list[str]:
        return []

    def validate_config(self, config):
        pass

    def process(self, df, context):
        return df


class TestMetricRegistry:
    """Tests for the MetricRegistry class."""

    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = MetricRegistry()
        registry.register(DummyMetric)
        assert registry.get("dummy") is DummyMetric

    def test_register_not_a_subclass(self):
        """Test that registering a non-BaseMetric class raises TypeError."""
        registry = MetricRegistry()
        with pytest.raises(TypeError, match="BaseMetric"):
            registry.register(str)  # type: ignore

    def test_register_duplicate_same_class(self):
        """Test that re-registering the same class is idempotent."""
        registry = MetricRegistry()
        registry.register(DummyMetric)
        registry.register(DummyMetric)  # Should not raise
        assert registry.get("dummy") is DummyMetric

    def test_register_duplicate_different_class(self):
        """Test that registering a different class with same name raises."""
        registry = MetricRegistry()
        registry.register(DummyMetric)

        # Create a conflicting class with same NAME
        class ConflictMetric(DummyMetric):
            NAME = "dummy"

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ConflictMetric)

    def test_get_not_found(self):
        """Test that getting a non-existent metric raises KeyError."""
        registry = MetricRegistry()
        registry._entry_points_loaded = True  # Skip entry point loading
        with pytest.raises(KeyError, match="not found"):
            registry.get("nonexistent")

    def test_list_metrics(self):
        """Test listing all registered metrics."""
        registry = MetricRegistry()
        registry._entry_points_loaded = True
        registry.register(DummyMetric)
        registry.register(AnotherMetric)

        metrics = registry.list_metrics()
        assert metrics == ["another", "dummy"]  # Sorted

    def test_load_all(self):
        """Test batch loading of metrics."""
        registry = MetricRegistry()
        registry._entry_points_loaded = True
        registry.register(DummyMetric)
        registry.register(AnotherMetric)

        instances = registry.load_all(["dummy", "another"])
        assert len(instances) == 2
        assert isinstance(instances[0], DummyMetric)
        assert isinstance(instances[1], AnotherMetric)

    def test_name_derivation_from_class_name(self):
        """Test automatic name derivation from CamelCase class name."""

        class MyCustomSearchMetric(BaseMetric):
            def __init__(self, config=None):
                pass

            @property
            def name(self):
                return "my_custom_search"

            @property
            def version(self):
                return "1.0.0"

            @property
            def required_columns(self):
                return []

            def validate_config(self, config):
                pass

            def process(self, df, context):
                return df

        registry = MetricRegistry()
        registry._entry_points_loaded = True
        registry.register(MyCustomSearchMetric)

        # Should derive name from the instance's name property
        assert "my_custom_search" in registry.list_metrics()
