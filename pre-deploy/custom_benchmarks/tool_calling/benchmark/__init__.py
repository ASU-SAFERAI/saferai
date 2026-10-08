"""Modular tool-calling benchmark implementation.

Each concern lives in its own module:

* :mod:`config` — candidate/evaluator models, bounds, and the CLI parser
* :mod:`data` — scenario loading, eval-dataset construction, dry-run preview
* :mod:`execution` — candidate WebSocket execution into observations
* :mod:`evaluation` — staged evaluator lifecycle (``score_accuracy``)
* :mod:`reporting` — detail rows and per-model summaries
* :mod:`artifacts` — provenance, run manifest, and CSV/XLSX/JSON output
* :mod:`runner` — orchestration (``main``)
"""
