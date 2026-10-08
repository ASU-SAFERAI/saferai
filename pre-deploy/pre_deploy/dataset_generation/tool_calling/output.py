"""Output serialization for tool-calling evaluation.

Provides a standalone function to serialize generated scenarios to JSON.
"""

import json
import os
from pathlib import Path

from .scenarios import generate_scenarios

__all__ = ["write_output"]


def write_output(num_variants: int, output_path: str = None) -> str:
    """Serialize generated scenarios to a JSON file.

    Parameters
    ----------
    num_variants : int
        Number of variants per combination to generate.
    output_path : str, optional
        Path to write the output file. Defaults to
        `tool_calling_seed.json` in the same directory as this module.

    Returns
    -------
    str
        Absolute path of the written file.

    Raises
    ------
    OSError
        If the output file cannot be written due to a filesystem error.
    """
    if output_path is None:
        target = Path(__file__).parent / "tool_calling_seed.json"
    else:
        target = Path(output_path)

    target = target.resolve()

    # Ensure the output directory exists
    os.makedirs(target.parent, exist_ok=True)

    # Generate scenario data
    scenarios = generate_scenarios(num_variants)

    # Write to a temp file in the same directory, then rename to target.
    # This avoids leaving partial files on write failure.
    temp_path = target.with_suffix(".json.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)
            f.write("\n")
        # Remove existing target first to avoid Windows permission issues
        if target.exists():
            os.remove(str(target))
        os.replace(str(temp_path), str(target))
    except Exception:
        # Clean up partial temp file if it exists
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise

    return str(target)
