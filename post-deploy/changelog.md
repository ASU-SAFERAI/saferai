# Changelog

## post-deploy

| Version | Date       | Change Description                                                                 |
|---------|------------|------------------------------------------------------------------------------------|
| 0.1.0   | 2026-08-21 | Initial release of post-deploy                                                   |
|         |            | - Pluggable metric framework with BaseMetric ABC and MetricRegistry                |
|         |            | - 4 built-in metrics: keyword_search, pii_search, zero_shot, llm_classifier       |
|         |            | - Pipeline orchestrator with YAML config and CLI                                   |
|         |            | - I/O layer: local CSV, S3, PostgreSQL                                             |
|         |            | - OpenAI-compatible LLM client for llm_classifier                                  |
|         |            | - SAFER preset with 14 keyword groups, 7 zero-shot labels, LLM prompt templates    |
|         |            | - Post-processing: configurable wide/long format output                            |
|         |            | - 68 unit + integration tests                                                      |
|         |            | - Python 3.11, uv for dependency management                                       |

_Version format: `MAJOR.MINOR.PATCH`_
