import sys, inspect
import json
import logging, os

_error_only_logs = ("dynamodb", "boto", "boto3", "botocore")


def setup_logging(log_level, local=False):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s"
        if local
        else "[%(levelname)s] %(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    for source in _error_only_logs:
        logging.getLogger(source).setLevel(logging.ERROR)

class LogEventMeta(type):
    _instances = {}

    def __call__(cls):
        file_name = os.path.basename(inspect.stack()[1].filename)
        local = "log_level" not in os.environ

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
        log_dict = json.loads(Encoder().encode(kwargs))
        log_dict["file"] = self.file

        if LogEvent.query_id is not None:
            log_dict["query_id"] = LogEvent.query_id
        if LogEvent.request_id is not None:
            log_dict["request_id"] = LogEvent.request_id
        return f"{message} {json.dumps(log_dict, indent=2 if self.local else None)}"


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return tuple(o)
        elif isinstance(o, str):
            return o.encode("unicode_escape").decode("ascii")
        elif isinstance(o, object):
            return o.__repr__()
        return super().default(o)
