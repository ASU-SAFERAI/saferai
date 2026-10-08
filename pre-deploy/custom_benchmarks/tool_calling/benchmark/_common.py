"""Shared imports, project path bootstrap, logger, and logging convention.

Centralizing these avoids repeating the ``sys.path`` bootstrap and the module
logger across the concern modules, and gives every module a single, consistent
source for the project root, the canonical dataset path, and the logging setup.

Every module in the tool-calling benchmark uses the shared ``logger`` here, and
every CLI entry point configures logging through :func:`configure_logging` so
the logger name, level handling, and record format stay identical across the
package.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is importable whether the benchmark is invoked as a
# script or imported as a package module.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The single logger name and record format shared by every entry point and
# module in the benchmark. Keeping both here means new modules only import
# ``logger`` and new CLIs only call ``configure_logging`` to match convention.
LOGGER_NAME = "tool_calling_benchmark"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

logger = logging.getLogger(LOGGER_NAME)


def configure_logging(level: str | int = "INFO") -> None:
    """Apply the shared logging convention for a CLI entry point.

    ``level`` may be a logging level name (case-insensitive, e.g. ``"DEBUG"``)
    or a numeric level. An unrecognized name falls back to ``INFO`` so a bad
    ``--log-level`` value never crashes an entry point.
    """
    if isinstance(level, str):
        resolved = getattr(logging, level.upper(), logging.INFO)
    else:
        resolved = level
    logging.basicConfig(level=resolved, format=LOG_FORMAT)


_DATASET_PATH = (
    _PROJECT_ROOT
    / "custom_benchmarks"
    / "tool_calling"
    / "tool_calling_seed_60_populated_multihop.json"
)

__all__ = [
    "_PROJECT_ROOT",
    "_DATASET_PATH",
    "LOGGER_NAME",
    "LOG_FORMAT",
    "logger",
    "configure_logging",
]
