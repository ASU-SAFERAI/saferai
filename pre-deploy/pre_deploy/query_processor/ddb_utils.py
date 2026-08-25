from decimal import Decimal
from typing import Union
import json


class Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            # Convert Decimal to float if it has decimals, otherwise to int
            return float(o) if o % 1 else int(o)
        elif isinstance(o, set):
            return tuple(o)
        elif isinstance(o, str):
            return o.encode("unicode_escape").decode("ascii")
        elif isinstance(o, object):
            return o.__repr__()
        return super().default(o)


def format_numeric_data(val: Decimal) -> Union[int, float]:
    if float(val) % 1 == 0:
        return int(val)
    return float(val)


def format_ddb_data(item):
    for k, v in item.items():
        if isinstance(v, dict):
            format_ddb_data(v)
        elif isinstance(v, list):
            for i in range(len(v)):
                if isinstance(v[i], dict):
                    format_ddb_data(v[i])
                elif isinstance(v[i], Decimal):
                    v[i] = format_numeric_data(v[i])
        elif isinstance(v, Decimal):
            item[k] = format_numeric_data(v)


def format_data_for_ddb(item):
    if isinstance(item, dict):
        for k, v in item.items():
            item[k] = format_data_for_ddb(v)
    elif isinstance(item, list):
        return [format_data_for_ddb(v) for v in item]
    elif isinstance(item, float):
        return Decimal(str(item))
    return item
