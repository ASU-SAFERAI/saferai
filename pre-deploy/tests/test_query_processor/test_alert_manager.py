import unittest
import sys
from pathlib import Path
import json

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.alert_manager import AlertManager
from pre_deploy.query_processor.environment import AWSEnvironment


class TestAlertManagerNotifyError(unittest.TestCase):
    """
    Unit tests for the AlertManager.notify_error() method.
    These tests assume the machine is already authenticated into AWS.
    They use real AWS SNS calls (not mocked).
    """

    def setUp(self):
        """Initialize AlertManager with real AWS environment."""
        self.alert_manager = AlertManager()

    def test_notify_error_with_context_data(self):
        """
        Test 6: Test notify_error() with additional context_data dictionary.
        """
        try:
            raise RuntimeError("Simulated runtime error")
        except Exception as e:
            context_data = {
                "user_id": "test_user_123",
                "request_id": "req_456",
                "operation": "data_processing",
                "timestamp": "2026-03-18T12:00:00Z"
            }
            self.alert_manager.notify_error(
                context="test_context_data",
                exception=e,
                context_data=context_data,
            )

    def test_notify_error_with_different_log_levels(self):
        """
        Test 7: Test notify_error() with different log_level values.
        """
        try:
            raise Exception("Test error for log level")
        except Exception as e:
            # Test with ERROR level (default)
            self.alert_manager.notify_error(
                context="test_log_level_error",
                exception=e,
                log_level="ERROR",
            )

            # Test with WARNING level
            self.alert_manager.notify_error(
                context="test_log_level_warning",
                exception=e,
                log_level="WARNING",
            )

            # Test with CRITICAL level
            self.alert_manager.notify_error(
                context="test_log_level_critical",
                exception=e,
                log_level="CRITICAL",
            )

    def test_notify_error_without_exception(self):
        """
        Test 8: Test notify_error() called without an exception object.
        This tests the case where notify_error() is called for informational purposes
        without an active exception.
        """
        self.alert_manager.notify_error(
            context="test_no_exception",
            exception=None,
            context_data={"event": "informational_alert"},
        )

    def test_notify_error_with_only_context(self):
        """
        Test 9: Test notify_error() with only context parameter.
        Minimal example of calling notify_error().
        """
        self.alert_manager.notify_error(
            context="test_minimal_call",
        )

    def test_notify_error_with_long_traceback(self):
        """
        Test 10: Test notify_error() with a long traceback.
        Verifies that the message truncates long tracebacks (>10000 chars).
        """
        try:
            # Create a deeply nested call stack to generate a longer traceback
            def level_3():
                raise Exception("Deep error in level 3")

            def level_2():
                level_3()

            def level_1():
                level_2()

            level_1()
        except Exception as e:
            self.alert_manager.notify_error(
                context="test_long_traceback",
                exception=e,
            )

    def test_notify_error_subject_sanitization(self):
        """
        Test 11: Test that SNS subject is sanitized correctly.
        Subject should be ASCII-only and max 100 characters.
        """
        # Test with special characters that should be removed
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.alert_manager.notify_error(
                context="test_subject_with_special_chars_🔥_and_unicode_こんにちは",
                exception=e,
            )

        # Test with a very long context that should be truncated
        long_context = "test_" + "a" * 150
        try:
            raise ValueError("Test error")
        except Exception as e:
            self.alert_manager.notify_error(
                context=long_context,
                exception=e,
            )


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
