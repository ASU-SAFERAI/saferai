from typing import Optional, Any, Type
from enum import Enum
import os


class ValidationUtilities:
    """Base class for validation utilities."""

    @staticmethod
    def _validate_id(id: str, field: str) -> str:
        """Validate that the ID is a non-empty string."""
        if not (isinstance(id, str) and id.strip()):
            raise ValueError(f"ID `{field}` must be a non-empty string.")
        return id

    @staticmethod
    def _validate_dict(data: dict) -> dict:
        """Validate that the data is a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Data must be a dictionary.")
        return data

    @staticmethod
    def _validate_keys_exist(data: dict, keys: list) -> None:
        """Validate that all required keys exist in the dictionary."""
        for key in keys:
            if key not in data:
                raise KeyError(f"Missing required key: '{key}'")

    @staticmethod
    def _validate_sequence(sequence: int, field: str) -> int:
        """Validate that sequence is a non-negative integer."""
        if not (isinstance(sequence, int) and sequence >= 0):
            raise ValueError(f"Sequence `{field}` must be a non-negative integer.")
        return sequence

    @staticmethod
    def _validate_type(obj: Any, expected_type: type, field: str) -> Any:
        """Validate that obj is of the expected type."""
        if not isinstance(obj, expected_type):
            raise TypeError(f"`{field}`: Expected type {expected_type.__name__}, got {type(obj).__name__}.")
        return obj

    @staticmethod
    def _validate_enum(value: str, enum_class: Type[Enum], field: str) -> str:
        """Validate that the value is a member of the given Enum class."""
        if value not in enum_class._value2member_map_:
            raise ValueError(
                f"`{field}`: Invalid value '{value}'. Must be one of: {list(enum_class._value2member_map_.keys())}"
            )
        return value

    @staticmethod
    def _validate_uri(uri: Optional[str], field: str) -> Optional[str]:
        """Validate that URI is a valid URL, file path, or None."""
        if not (uri is None or isinstance(uri, str)):
            raise ValueError(f"`{field}`: URI must be a string or None.")
        if uri is None:
            return None

        # Check for valid URL prefixes or existing file paths
        valid_prefixes = ('http://', 'https://', 'ftp://', 's3://', 'file://')
        has_valid_prefix = uri.startswith(valid_prefixes)
        file_exists = os.path.exists(uri)

        if not (has_valid_prefix or file_exists):
            raise ValueError(
                f"`{field}`: URI '{uri}' must start with one of {valid_prefixes} or point to an existing file."
            )

        return uri
