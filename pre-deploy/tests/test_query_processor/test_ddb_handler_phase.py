import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock

sys.path.append(Path(__file__).resolve().parent.parent.parent.as_posix())

from pre_deploy.query_processor.ddb_handler import (
    read_model_config_ddb_table,
    read_responses_from_query_processor_table,
)
from pre_deploy.query_processor.alert_manager import AlertManager
from pre_deploy.query_processor.environment import AWSEnvironment


class TestDdbHandlerPhaseReads(unittest.TestCase):
    def test_read_responses_returns_empty_dict_when_no_items(self):
        table = Mock()
        table.query.return_value = {"Items": []}

        dynamodb = Mock()
        dynamodb.Table.return_value = table

        session = Mock()
        session.resource.return_value = dynamodb

        environment = Mock()
        environment.session = session
        environment.region = "us-west-2"
        environment.query_processor_table = "query_processor_table"

        alert_manager = Mock()

        responses = read_responses_from_query_processor_table(
            environment=environment,
            run_id="run-1",
            metric_phase="phase-1",
            alert_manager=alert_manager,
        )

        self.assertEqual(responses, {})

    def test_read_model_config_scan_live_aws(self):
        # Uses live AWS credentials/session. Run after SSO login.
        environment = AWSEnvironment(
            target_account_id=None,
            role_name=None
        )
        alert_manager = AlertManager(aws_environment=environment)
        items = read_model_config_ddb_table(
            environment=environment,
            alert_manager=alert_manager,
        )
        self.assertIsInstance(items, list)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
