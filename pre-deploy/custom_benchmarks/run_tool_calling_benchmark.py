#!/usr/bin/env python
"""Run the tool-calling benchmark.

This is the executable entry point and the backward-compatible import surface.
The implementation is split by concern under ``custom_benchmarks.tool_calling``:

* ``benchmark.config`` — candidate/evaluator models, bounds, and CLI parser
* ``benchmark.data`` — scenario loading, eval-dataset construction, dry-run
* ``benchmark.execution`` — candidate WebSocket execution into observations
* ``benchmark.evaluation`` — staged evaluator lifecycle (``score_accuracy``)
* ``benchmark.reporting`` — detail/summary reports, provenance, artifacts
* ``benchmark.runner`` — orchestration (``main``)

Candidate generation goes through the agentic WebSocket only: ``execution`` uses
``AgenticWebSocketClient`` and invokes ``client.query(request)`` per scenario.
There is no Query Processor candidate-generation path here.

Existing callers and tests import both public and private helpers from this
module and monkeypatch them here, so every concern symbol is re-exported and
patches are routed to the module that actually consults them.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when this file is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from custom_benchmarks.tool_calling.benchmark import (
    artifacts as _artifacts,
    config as _config,
    data as _data,
    evaluation as _evaluation,
    execution as _execution,
    persistence as _persistence,
    reporting as _reporting,
    runner as _runner,
)
from custom_benchmarks.tool_calling.agentic.client import AgenticWebSocketClient
from custom_benchmarks.tool_calling.agentic.config import (
    load_credentials,
    redact_sensitive_data,
    redact_sensitive_text,
    validate_websocket_timeout,
)

# Re-export every public and private helper from the concern modules so the
# historical flat import surface (custom_benchmarks.run_tool_calling_benchmark)
# keeps working without a flag-day migration.
_CONCERN_MODULES = (
    _config, _data, _execution, _evaluation, _persistence, _reporting, _artifacts
)
for _module in _CONCERN_MODULES:
    for _name in dir(_module):
        if not _name.startswith("__"):
            globals()[_name] = getattr(_module, _name)

# The Redshift data-mart write lives in the ``data_mart`` subpackage rather than
# a flat concern module; re-export its entry point so the runner honors a
# facade-level patch (and tests can mock it) like the other collaborators.
from custom_benchmarks.tool_calling.benchmark.data_mart import (  # noqa: E402
    write_benchmark_run_to_data_mart,
)

# Names read as module globals by score_accuracy; tests patch them here.
_EVALUATION_PATCH_NAMES = (
    "start_tool_call_failure_modes",
    "finalize_tool_call_failure_modes",
    "sleep",
    "POLL_INTERVAL_SECONDS",
    "MAX_POLLS",
)


def _sync_patches(module, names):
    """Copy patched facade names into ``module`` and return their originals."""
    previous = {name: getattr(module, name) for name in names if hasattr(module, name)}
    for name in names:
        if name in globals() and hasattr(module, name):
            setattr(module, name, globals()[name])
    return previous


def _restore(module, previous):
    for name, value in previous.items():
        setattr(module, name, value)


def score_accuracy(*args, **kwargs):
    """Delegate to the evaluation module, honoring facade-level monkeypatches."""
    previous = _sync_patches(_evaluation, _EVALUATION_PATCH_NAMES)
    try:
        return _evaluation.score_accuracy(*args, **kwargs)
    finally:
        _restore(_evaluation, previous)


_score_accuracy_facade = score_accuracy


def main() -> None:
    """Run orchestration after routing facade-level patches to their modules.

    ``main`` collaborators (load_credentials, AgenticWebSocketClient,
    collect_candidate_responses, build_trace_eval_dataset, score_accuracy,
    build_report, write_outputs) are resolved from ``runner``'s namespace, so
    patches applied to this facade are copied there for the duration of the run.
    Evaluation globals are copied into the evaluation module so a patched
    score_accuracy is unnecessary, but a directly patched score_accuracy is
    still honored.
    """
    runner_names = [
        name
        for name in dir(_runner)
        if not name.startswith("__") and name in globals() and name != "main"
    ]
    runner_prev = {name: getattr(_runner, name) for name in runner_names}
    eval_prev = _sync_patches(_evaluation, _EVALUATION_PATCH_NAMES)
    try:
        for name in runner_names:
            # Route a bare facade score_accuracy through the evaluation module
            # rather than installing the recursive facade wrapper into runner.
            if name == "score_accuracy" and globals()[name] is _score_accuracy_facade:
                setattr(_runner, name, _evaluation.score_accuracy)
            else:
                setattr(_runner, name, globals()[name])
        _runner.main()
    finally:
        _restore(_runner, runner_prev)
        _restore(_evaluation, eval_prev)


if __name__ == "__main__":
    main()
