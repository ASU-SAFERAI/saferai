import sys
import inspect
import json
import logging
import os
from os.path import basename

__all__ = ["setup_logging", "LogEvent"]

_error_only_logs = (
    "boto",
    "boto3",
    "botocore",
    "urllib3",
    "s3transfer",
    "nose",
    "bcdocs",
    "openai",
    "httpx",
    "vertexai",
    "google",
    "httpcore",
    "opensearch",
    "ddtrace.tracer",
    "ddtrace.llmobs._llmobs",
    "ddtrace",
    "dspy",
    "LiteLLM",
)

omit_keys = {
    "asurite",
    "query",
    "response",
    "system_prompt",
    "results",
    "search_results",
    "final_params",
}


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return tuple(o)
        elif isinstance(o, str):
            return o.encode("unicode_escape").decode("ascii")
        elif isinstance(o, object):
            return o.__repr__()
        return super().default(o)


def setup_logging(log_level, local=False):
    logging.basicConfig(
        format=(
            "%(asctime)s [%(levelname)s] %(message)s"
            if local
            else "[%(levelname)s] %(message)s"
        ),
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    for source in _error_only_logs:
        logging.getLogger(source).setLevel(logging.ERROR)

    # print_log_tree()  # Call this here to see the loggers after setup, only call if needed since this will print in cloudwatch.


def print_log_tree():
    logger_dict = logging.root.manager.loggerDict
    for logger_name in logger_dict:
        if isinstance(logger_dict[logger_name], logging.Logger):
            logger = logging.getLogger(logger_name)
            print(
                f"Logger: '{logger_name}', Level: {logging.getLevelName(logger.level)}"
            )


class LogEventMeta(type):
    _instances = {}

    def __call__(cls):
        file_name = os.path.basename(inspect.stack()[1].filename)
        local = "log_level" not in os.environ

        # singleton - only one instance per file
        if file_name not in cls._instances:
            cls._instances[file_name] = super().__call__(file=file_name, local=local)

        return cls._instances[file_name]


class LogEvent(metaclass=LogEventMeta):
    query_id = None
    request_id = None

    def __init__(self, file, local):
        self.file = file
        self.local = local

    def format(self, message, *args, **kwargs):

        def omit_specified_keys(data, keys_to_omit):
            if isinstance(data, dict):
                new_dict = {}
                for k, v in data.items():
                    if k not in keys_to_omit:
                        new_dict[k] = omit_specified_keys(v, keys_to_omit)
                return new_dict
            elif isinstance(data, list):
                return [omit_specified_keys(item, keys_to_omit) for item in data]
            else:
                return data

        def truncate_large_values(d):
            truncate_value = 10_000
            if isinstance(d, dict):
                for key, value in d.items():
                    d[key] = truncate_large_values(value)
            elif isinstance(d, list):
                d = [truncate_large_values(item) for item in d]
            # elif isinstance(d, (int, float)) and d > truncate_value:
            #     return truncate_value
            elif isinstance(d, str) and len(d) > truncate_value:
                return d[:truncate_value] + "..."
            return d

        log_dict = json.loads(Encoder().encode(kwargs))
        log_dict["file"] = self.file

        if LogEvent.query_id is not None:
            log_dict["query_id"] = LogEvent.query_id
        if LogEvent.request_id is not None:
            log_dict["request_id"] = LogEvent.request_id

        # Remove specified keys if log level is INFO
        if logging.getLogger().level == logging.INFO:
            log_dict = omit_specified_keys(log_dict, omit_keys)

        # if logging.getLogger().level == logging.DEBUG or logging.getLogger().level == logging.INFO:
        # Truncate large values in the dictionary
        log_dict = truncate_large_values(log_dict)

        return f"{message} {json.dumps(log_dict, indent=2 if self.local else None)}"
