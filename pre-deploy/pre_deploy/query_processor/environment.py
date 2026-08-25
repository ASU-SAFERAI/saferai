import os
from typing import Optional


class AWSEnvironment:
    """
    A templated class to handle cross-account AWS environment setup for the pre_deploy Query Processor.
    This class assumes you are already authenticated into a given AWS account.
    It checks if the current account matches the target account ID for pre_deploy operations.
    """

    def __init__(self, target_account_id: Optional[str], role_name: Optional[str], region: str = "us-west-2"):
        """
        :param target_account_id: Populate with a string account ID if different from your default environment. Keep as `None` if using the same account.
        :type target_account_id: str
        :param role_name: Populate with the name of your AWS account role for cross-account access. Keep as `None` if using the same account.
        :type role_name: str
        :param region: The name of the region to use for the cross-account session. Default is "us-west-2".
        :type region: str
        """
        import logging
        import boto3

        self.logger = logging.getLogger(__name__)
        self.target_account_id = target_account_id
        self.region = region

        # Initialize STS client - this establishes the identity of the current AWS account
        self.sts_client = boto3.client("sts", region_name=self.region)
        identity = self.sts_client.get_caller_identity()
        self.current_account_id = identity["Account"]

        # Determine if cross-account access is needed
        if self.target_account_id is None:
            self.target_account_id = self.current_account_id
        self.use_cross_account = self.current_account_id != self.target_account_id

        # Only gets populated if cross-account access is needed, otherwise invalid
        self.cross_account_role_arn = f"arn:aws:iam::{self.target_account_id}:role/{role_name}"

        self.session = self._get_boto3_session()

        # AWS clients using the appropriate session
        self.sqs_client = self.session.client('sqs', region_name=self.region)

        # Environment variables
        self.sqs_queue = os.environ.get("query_processor_queue", "query-processor-dev")
        self.sqs_queue_url = (f"https://sqs.{self.region}.amazonaws.com/"
                              f"{self.target_account_id}/{self.sqs_queue}")
        self.query_processor_table = os.environ.get("query_processor_table", "query_processor_dev")
        self.stats_table = os.environ.get("stats_table", "stats_dev")
        self.model_config_table = os.environ.get("model_config", "model_config_dev")
        self.log_level = os.environ.get("log_level", "DEBUG")
        self.local_run = "log_level" not in os.environ

        # SNS monitoring topic for error reporting and alerts.
        self.sns_topic_arn = os.environ.get("sns_alert_topic", f"arn:aws:sns:{self.region}:{self.target_account_id}:alerts-dev")
        self.sns_client = self.session.client('sns', region_name=self.region)

    def _get_boto3_session(self):
        """
        Returns a boto3 session, assuming cross-account role if needed.

        Returns:
            tuple: (boto3.Session, account_id)
        """
        import boto3
        if self.use_cross_account:
            self.logger.info(f"Current account ID differs from target account ID. Assuming cross-account role.")

            try:
                assumed_role = self.sts_client.assume_role(
                    RoleArn=self.cross_account_role_arn,
                    RoleSessionName="CrossAccountSession"
                )

                credentials = assumed_role['Credentials']

                session = boto3.Session(
                    aws_access_key_id=credentials['AccessKeyId'],
                    aws_secret_access_key=credentials['SecretAccessKey'],
                    aws_session_token=credentials['SessionToken'],
                    region_name=self.region
                )

                self.logger.debug(f"Successfully assumed role in account {self.target_account_id}")
                return session

            except Exception as e:
                self.logger.error(f"Failed to assume cross-account role: {str(e)}")
                raise Exception(f"Cross-account role assumption failed: {str(e)}")
        else:
            # self.logger.info(f"Using current account {self.current_account_id}")
            return boto3.Session(region_name=self.region)
