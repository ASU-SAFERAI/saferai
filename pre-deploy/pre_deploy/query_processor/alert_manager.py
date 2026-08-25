import traceback
from typing import Optional
import json
import logging

from .environment import AWSEnvironment

logger = logging.getLogger(__name__)


class AlertManager:
    def __init__(self, aws_environment: Optional[AWSEnvironment] = None):
        if aws_environment is not None:
            self.aws_environment = aws_environment
        else:
            self.aws_environment = AWSEnvironment(
                target_account_id=None,
                role_name=None,
            )

    def notify_error(
        self,
        context: str,
        exception: Exception = None,
        context_data: dict = None,
        log_level: str = "ERROR",
    ):
        """
        Centralized SNS error reporting.

        Args:
            context (str): Location or function name where error occurred.
            exception (Exception): The exception object, if available.
            context_data (dict): Additional context data (user info, method, etc.).
            log_level (str): Log level for the error message.
        """

        try:
            message_data = dict(context_data or {})
            message_data["context"] = context
            message_data["log_level"] = log_level

            if exception is not None:
                tb_text = "".join(
                    traceback.format_exception(
                        type(exception),
                        exception,
                        exception.__traceback__,
                    )
                )
                message_data["exception_type"] = type(exception).__name__
                message_data["exception_message"] = str(exception)
            else:
                tb_text = traceback.format_exc()

            if tb_text and tb_text != "NoneType: None\n":
                message_data["traceback"] = (
                    tb_text[:10000] + "..." if len(tb_text) > 10000 else tb_text
                )

            self.publish_sns_message(message_data, f"{log_level} in {context}")

        except Exception as e:
            logger.warning(
                "Failed to send error notification for context '%s': %s",
                context,
                e,
                exc_info=True,
            )

    def publish_sns_message(self, sns_message, subject):
        # Ensure SNS Subject complies with AWS constraints: ASCII-only, max length 100.
        if subject is None:
            safe_subject = "Alert"
        else:
            safe_subject = str(subject)
            # Remove non-ASCII characters.
            safe_subject = safe_subject.encode("ascii", "ignore").decode("ascii")
            if not safe_subject:
                safe_subject = "Alert"
        if len(safe_subject) > 100:
            safe_subject = safe_subject[:100]

        self.aws_environment.sns_client.publish(
            TopicArn=self.aws_environment.sns_topic_arn,
            Message=json.dumps(sns_message, indent=4),
            Subject=safe_subject,
        )
