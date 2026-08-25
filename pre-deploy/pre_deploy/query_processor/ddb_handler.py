import boto3
from boto3.dynamodb.conditions import Key
import logging
import time
from .logs import LogEvent
from .ddb_utils import format_ddb_data, format_data_for_ddb
from .environment import AWSEnvironment
from .alert_manager import AlertManager

# Set up logging
logger = logging.getLogger(__name__)
log_event = LogEvent()


def write_to_query_processor_table(environment: AWSEnvironment, item: dict, alert_manager: AlertManager):
    try:
        session = environment.session
        table_name = environment.query_processor_table
        region = environment.region
        dynamodb = session.resource("dynamodb", region_name=region)
        table = dynamodb.Table(table_name)

        item['time_started'] = int(time.time())
        formatted_item = format_data_for_ddb(item)
        logger.debug(log_event.format("Write_to_DDB",
                                      table_name=table_name,
                                      item=formatted_item))
        table.put_item(Item=formatted_item)
        logger.debug(log_event.format("Write_to_DDB_Success",
                                      table_name=table_name,
                                      item_id=item.get("id")))
    except Exception as e:
        logger.error(log_event.format("Write_to_DDB_Failure",
                                      table_name=table_name,
                                      error=str(e)))
        alert_manager.notify_error(
            context="write_to_query_processor_table",
            exception=e,
            context_data={"table_name": table_name, "item_id": item.get("id")},
            log_level="ERROR"
        )
        raise


def read_responses_from_query_processor_table(environment: AWSEnvironment, run_id: str, metric_phase: str,
                                              alert_manager: AlertManager) -> dict:
    """
    Fetches model responses from DynamoDB by run_id and metric_phase.
    Args:
        run_id (str): The run ID to query the responses for.
        metric_phase (str): The metric phase to filter the responses.
    Returns:
        dict: A dictionary containing the model responses indexed by query number.
    """
    try:
        region = environment.region
        query_processor_table = environment.query_processor_table

        logger.debug(log_event.format("Fetching_Model_Responses",
                                    run_id=run_id,
                                    metric_phase=metric_phase,
                                    table_name=query_processor_table))

        dynamodb = environment.session.resource('dynamodb', region_name=region)
        table = dynamodb.Table(query_processor_table)
        response = table.query(
            IndexName='run_id_metric_phase',
            KeyConditionExpression=(Key('run_id').eq(run_id) &
                                    Key('metric_phase').eq(metric_phase))
        )
        items = response.get('Items')
        while 'LastEvaluatedKey' in response:
            response = table.query(IndexName='run_id_metric_phase',
                                   KeyConditionExpression=(Key('run_id').eq(run_id) &
                                                           Key('metric_phase').eq(metric_phase)),
                                   ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response['Items'])

        logger.debug(log_event.format("DDB_Query_Success", run_id=run_id, metric_phase=metric_phase))
    except Exception as e:
        logger.error(log_event.format("DDB_Query_Failure", run_id=run_id, metric_phase=metric_phase, error=str(e)))
        alert_manager.notify_error(
            context="read_responses_from_query_processor_table",
            exception=e,
            context_data={"run_id": run_id, "metric_phase": metric_phase},
            log_level="ERROR"
        )
        raise

    if not items:
        logger.warning(log_event.format("No_Items_Found", run_id=run_id, metric_phase=metric_phase))
        return {}

    results = {}
    for item in items:
        query_number = item.get('query_number')
        if str(query_number).isdigit():
            query_number = str(int(query_number))
        else:
            query_number = str(query_number)

        model_response = item.get('model_response')
        results[query_number] = model_response
        logger.debug(log_event.format("DDB_Item_Processed", query_number=query_number, run_id=run_id, metric_phase=metric_phase))

    return results


def read_model_config_ddb_table(environment: AWSEnvironment, alert_manager: AlertManager) -> dict:
    try:
        session = environment.session
        region = environment.region
        model_config_table = environment.model_config_table
        dynamodb = session.resource('dynamodb', region_name=region)

        table = dynamodb.Table(model_config_table)
        response = table.scan()

        items = response.get('Items', [])

        while "LastEvaluatedKey" in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        for item in items:
            format_ddb_data(item)

        return items
    except Exception as e:
        logger.error(
            log_event.format(
                "Read_Model_Config_DDB_Failure",
                error=str(e),
            )
        )
        alert_manager.notify_error(
            context="read_model_config_ddb_table",
            exception=e,
            log_level="ERROR"
        )
        raise
