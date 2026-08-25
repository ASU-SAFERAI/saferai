# SAFER AI

**System for AI Feedback, Evaluation, and Reporting**

SAFER AI is an open-source, model-agnostic platform for evaluating, monitoring, and improving AI applications across the full deployment lifecycle. Developed by the AI Acceleration team at Arizona State University, SAFER AI operates in production across ASU's digital environments, including within the AI Research (AIR) environment, where it serves as the trust and safety layer for AI experiences built by researchers, faculty, and staff.

The suite combines three core capabilities in a single feedback loop:

1. **Pre-deployment evaluation** — automated assessments of accuracy, bias, fairness, robustness, and safety-critical response quality against curated scenario datasets
2. **Real-time detection** — small language models acting as monitoring agents that classify concerning interactions (prompt injection, content safety violations) as they occur
3. **Post-deployment analytics** — interaction pattern review and summarization that surfaces risks, engagement quality trends, and unmet needs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        SAFER AI                             │
├───────────────────┬───────────────────┬─────────────────────┤
│   pre-deploy/     │  in-production/   │   post-deploy/      │
│                   │                   │                     │
│  Evaluation       │  Real-time        │  Post-deployment    │
│  harness with     │  detection via    │  metric framework   │
│  bias, fairness,  │  Prompt-Guard     │  for keyword,       │
│  robustness, and  │  classifier and   │  PII, zero-shot,    │
│  custom metrics   │  content          │  and LLM-based      │
│                   │  moderation       │  classification     │
└───────────────────┴───────────────────┴─────────────────────┘
```

## Components

### [`pre-deploy/`](pre-deploy/) — Pre-Deployment Evaluation

A framework for comprehensive LLM evaluation, providing reusable evaluation steps, perturbation strategies, and scoring pipelines. Runs automated assessments before any AI tool reaches users.

**Capabilities:**
- Bias benchmarking (BBQ — Bias Benchmark for Question-Answering)
- Reading comprehension evaluation (BoolQ)
- Context-aware responsible evaluation (CARE)
- Multi-dimensional reasoning and bias assessment (MARBLE)
- Consistency and robustness scoring
- Custom GEval with configurable evaluation steps
- DAG-based deterministic evaluations (fact-checking, etc.)
- Perturbation strategies for stress-testing

**Key features:**
- Model-agnostic via adapter interfaces (OpenAI, open-source models, custom providers)
- Batch query processing via SQS + DynamoDB backend
- Standard input dataset format supporting single-turn and multi-turn conversations
- Built-in prompt templates for educational AI evaluation (Socratic scoring, direct-answer avoidance, tone)

```bash
pip install "git+ssh://git@github.com/<your-org>/saferai.git@main#subdirectory=pre-deploy"
```

### [`in-production/`](in-production/) — Real-Time Detection

Prompt-injection and content-safety scoring deployed as an AWS Lambda function. Classifies high-stakes interactions as they occur and alerts response teams.

**Metrics:**

| Metric | Detection | Model |
|--------|-----------|-------|
| `promptguard` | Prompt injection / jailbreak | Prompt-Guard-2 (DeBERTa-based), runs in-container |
| `content_moderation` | Content safety categories | OpenAI omni-moderation endpoint |

**Key features:**
- Containerized with model weights cached in S3 for fast cold starts
- Configurable alerting via SNS topics (self-harm, prompt injection, general safety)
- DynamoDB model registry with threshold-based alerting
- Supports both Prompt-Guard v1 (3-class) and v2 (2-class)

```bash
# Local testing
cd in-production
python test_main.py --test apigw-event
```

### [`post-deploy/`](post-deploy/) — Post-Deployment Analytics

A pluggable metric framework for evaluating interaction data at scale. Converts monitored usage into actionable insights through keyword detection, PII scanning, zero-shot classification, and LLM-based analysis.

**Built-in metrics:**
- **keyword_search** — Regex-based keyword matching (ships with 14 SAFER keyword groups)
- **pii_search** — PII detection via Microsoft Presidio
- **zero_shot** — Zero-shot NLI classification using DeBERTa models
- **llm_classifier** — Structured prompt classification via any OpenAI-compatible API

**Key features:**
- Pipeline architecture for chaining metrics with configurable I/O
- CLI interface for running evaluations from YAML configs
- SAFER preset with pre-configured safety-focused prompts and keyword groups
- Extensible via entry-point plugin system
- Compatible with OpenAI, Azure OpenAI, Ollama, vLLM, or any `/v1/chat/completions` endpoint

```bash
cd post-deploy
uv pip install -e ".[all]"
post-deploy run --preset safer --file-paths data/sample.csv
```

## Design Principles

- **Model-agnostic** — Works with commercial, open-source, and custom AI models through documented adapter interfaces. Support for new models is added by defining new adapters rather than modifying core logic.
- **Infrastructure-independent** — Deploys on any institution's preferred stack; no hard-coded integrations.
- **Modular and replaceable** — Internal models and tools are interchangeable as the AI safety landscape evolves.
- **Institution-configurable** — Risk categories, escalation thresholds, and detection sensitivity are configurable per deployment.
- **Privacy-preserving** — Designed to work with synthetic and de-identified data; no identifiable student records in the repository.

## Requirements

- Python >= 3.11 (pre-deploy), >= 3.10 (post-deploy)
- See individual component READMEs for specific dependencies

## Quick Start

Each component is independently installable and runnable. See the README in each subdirectory for detailed setup instructions:

- [Pre-Deploy README](pre-deploy/README.md)
- [In-Production README](in-production/README.md)
- [Post-Deploy README](post-deploy/README.md)

## Repository Structure

```
saferai/
├── pre-deploy/              # Pre-deployment evaluation harness
│   ├── pre_deploy/          # Package: metrics, loaders, perturbation, DAGs
│   ├── tests/               # Unit tests
│   ├── pyproject.toml
│   └── README.md
├── in-production/           # Real-time detection (AWS Lambda)
│   ├── main.py              # Lambda handler
│   ├── prompt_guard.py      # Prompt-Guard classifier
│   ├── evals.py             # OpenAI content moderation
│   ├── Dockerfile
│   └── README.md
├── post-deploy/             # Post-deployment analytics framework
│   ├── src/post_deploy/     # Package: core, metrics, presets, CLI
│   ├── tests/               # Unit + integration tests
│   ├── examples/            # Demo scripts, notebooks, YAML configs
│   ├── pyproject.toml
│   └── README.md
└── README.md                # This file
```

## License

SAFER AI is released under an OSI-approved open-source license. See individual component directories for specific license information.

## About

SAFER AI is developed and maintained by the AI Acceleration team and Research Technology Office at Arizona State University. The platform operates in production across ASU's enterprise AI deployments, serving one of the largest and most diverse student bodies in the United States.

For more information, see the [ASU Enterprise Technology feature](https://tech.asu.edu/features/evaluation-framework-sets-new-benchmark-ethical-ai) on the evaluation framework.
