"""I/O layer: input sources and output managers."""

from .base import InputSource, OutputManager
from .local import LocalCSVInputSource, LocalOutputManager

__all__ = [
    "InputSource",
    "OutputManager",
    "LocalCSVInputSource",
    "LocalOutputManager",
]

# S3 backend is importable but not eagerly loaded
# to avoid requiring optional dependencies at import time.
# Use:
#   from post_deploy.io.s3 import S3InputSource, S3OutputManager
