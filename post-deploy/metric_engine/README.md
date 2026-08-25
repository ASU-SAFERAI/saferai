# metric-engine

A pluggable metric framework for evaluating text data. Ships with built-in metrics for keyword search, PII detection, zero-shot NLI classification, and LLM-based classification via any OpenAI-compatible API.

## Quick Start

### Setup

```bash
cd metric_engine

# Create venv with uv (recommended)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e ".[pii,ml,dev]"
python -m spacy download en_core_web_lg

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install -e ".[pii,ml,dev]"
python -m spacy download en_core_web_lg
```

### Run the Demo

```bash
```

### Run in a Notebook

Open `examples/demo_notebook.ipynb` in VS Code and select the `.venv` kernel.

### Run Tests

```bash
pytest tests/
```

### CLI Usage

```bash
metric-engine run --config examples/pipeline_with_llm.yaml
metric-engine run --preset safer --file-paths data/sample.csv
metric-engine list-metrics
metric-engine validate examples/pipeline_with_llm.yaml
```

## Metrics

### keyword_search

Regex-based keyword matching against text columns. Fast, no external dependencies.

Produces per keyword group:
- `{column}_{group}_found` (bool)
- `{column}_{group}_matches` (list of matched keywords)

```python
from metric_engine.core.metric import MetricContext
from metric_engine.metrics.keyword_search import KeywordSearchMetric

context = MetricContext(columns={"query": "query_raw", "response": "response_raw"})

metric = KeywordSearchMetric(config={
    "target_columns": ["query", "response"],
    "keyword_groups": {
        "confusion": ["I'm sorry", "don't understand", "clarify"],
        "error": ["error", "try again"],
        "prompt_override": ["ignore", "previous instructions"],
    },
})

result = metric.process(df, context)
```

YAML config:
```yaml
- name: keyword_search
  config:
    target_columns: [query, response]
    preset: safer  # loads 14 keyword groups from the built-in SAFER preset
```

### pii_search

Detects personally identifiable information using Microsoft Presidio.

Requires: `uv pip install -e ".[pii]"` and `python -m spacy download en_core_web_lg`

Produces per target column:
- `pii_{column}_any_found` (bool)
- `pii_{column}_found_all` (list)
- `pii_{column}_found_distinct` (list)
- `pii_{column}_entities_found` (list of entity types)
- `pii_{column}_ne_{entity_type}` (list, when verbose=True)

```python
from metric_engine.metrics.pii_search import PiiSearchMetric

metric = PiiSearchMetric(config={
    "target_columns": ["query"],
    "entity_types": ["EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER", "URL"],
    "language": "en",
    "verbose": True,
})

result = metric.process(df, context)
```

YAML config:
```yaml
- name: pii_search
  config:
    target_columns: [query]
    entity_types: [EMAIL_ADDRESS, PERSON, PHONE_NUMBER, URL]
    verbose: true
```

### zero_shot

Zero-shot NLI classification using a HuggingFace model. Each prompt defines a hypothesis template and produces float confidence scores per label.

Requires: `uv pip install -e ".[ml]"`

The SAFER preset includes 7 labels:

| Label | Target | Detects |
|-------|--------|---------|
| `q_question` | query | User is asking a question |
| `q_role` | query | Role/persona assignment |
| `q_reward` | query | Prize/reward social engineering |
| `q_punishment` | query | Punishment/threat social engineering |
| `q_prompt` | query | Prompt injection attempt |
| `r_confusion` | response | Bot needs clarification |
| `r_kb` | response | Info not in knowledge base |

```python
from metric_engine.metrics.zero_shot import ZeroShotMetric

metric = ZeroShotMetric(config={
    "target_columns": ["query", "response"],
    "model": "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "prompts": [
        {
            "hypothesis_template": "This text is {}",
            "target": "query",
            "labels": {"q_question": "asking a question."},
        },
        {
            "hypothesis_template": "This text has at least one sentence {}",
            "target": "query",
            "labels": {
                "q_role": "assigning a role, identity, or persona",
                "q_prompt": "telling you to change or ignore prior instructions or system prompts",
            },
        },
        {
            "hypothesis_template": "This text is stating that {}",
            "target": "response",
            "labels": {
                "r_confusion": "additional clarification is needed",
                "r_kb": "information is not listed in knowledge base.",
            },
        },
    ],
})

result = metric.process(df, context)
```

YAML config:
```yaml
- name: zero_shot
  config:
    target_columns: [query, response]
    preset: safer
    model: "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
```

To use a local model (faster, no download on each run):
```bash
python scripts/download_model.py --output-dir ./models/zeroshot_offline
```
Then set `model: "./models/zeroshot_offline"` in config.

### llm_classifier

Sends structured prompts to any OpenAI-compatible LLM API. Parses comma-separated label responses into validated boolean columns.

Requires: `uv pip install -e ".[llm]"`

Compatible endpoints: OpenAI, Azure OpenAI, Ollama, vLLM, or any `/v1/chat/completions` server.

The SAFER preset classifies:
- **Query**: `asking_a_question`, `other`
- **Response**: `not_in_knowledgebase`, `clarification_needed`, `other`

```python
from metric_engine.metrics.llm_classifier import LLMClassifierMetric

metric = LLMClassifierMetric(config={
    "target_columns": ["query", "response"],
    "preset": "safer",
    "llm_client": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "sk-...",
        "model": "gpt-4",
        "temperature": 0.0,
        "max_concurrent": 4,
    },
})

result = metric.process(df, context)
```

YAML config:
```yaml
- name: llm_classifier
  config:
    target_columns: [query, response]
    preset: safer
    llm_client:
      base_url: "https://api.openai.com/v1"  # or http://localhost:11434/v1 for Ollama
      api_key: "sk-..."
      model: "gpt-4"
      temperature: 0.0
      max_concurrent: 4
```

For testing without API calls, pass a mock client:
```python
from metric_engine.io.llm_client import BaseLLMClient

class MockClient(BaseLLMClient):
    def query(self, prompt):
        return "asking_a_question"

metric = LLMClassifierMetric(config={
    "target_columns": ["query"],
    "prompts": [{"target": "query", "prefix": "llm", "categories": ["asking_a_question", "other"], "template": "Classify: {text}"}],
    "client_instance": MockClient(),
})
```

## Pipeline

Chain multiple metrics together with configurable I/O and post-processing:

```python
from metric_engine.core.config import PipelineConfig, MetricConfig
from metric_engine.core.pipeline import Pipeline
from metric_engine.core.registry import MetricRegistry
from metric_engine.io.local import LocalCSVInputSource, LocalOutputManager
from metric_engine.metrics.keyword_search import KeywordSearchMetric

registry = MetricRegistry()
registry.register(KeywordSearchMetric)

config = PipelineConfig(
    input={"columns": {"query": "query_raw", "response": "response_raw"}},
    metrics=[
        MetricConfig(name="keyword_search", config={
            "target_columns": ["query"],
            "keyword_groups": {"override": ["ignore", "forget"]},
        }),
    ],
    output={"format": "wide"},
    post_process={"enabled": False},
)

pipeline = Pipeline(
    config=config,
    input_source=LocalCSVInputSource("data/input.csv"),
    output_manager=LocalOutputManager("outputs"),
    registry=registry,
)
pipeline.run()
```

See `examples/pipeline_with_llm.yaml` for a full YAML config example.

## Custom Metrics

Create your own by subclassing `BaseMetric`:

```python
from metric_engine.core.metric import BaseMetric, MetricContext

class TextLengthMetric(BaseMetric):
    NAME = "text_length"

    def __init__(self, config=None):
        self._config = config or {}
        self._target_columns = self._config.get("target_columns", ["query"])

    @property
    def name(self): return self.NAME
    @property
    def version(self): return "1.0.0"
    @property
    def required_columns(self): return self._target_columns

    def validate_config(self, config): pass

    def process(self, df, context):
        for col in self._target_columns:
            actual = context.get_column(col)
            df[f"{col}_word_count"] = df[actual].str.split().str.len()
        return df
```

Register via entry points in `pyproject.toml` for auto-discovery:
```toml
[project.entry-points."metric_engine.metrics"]
text_length = "my_package:TextLengthMetric"
```

## Project Structure

```
metric_engine/
├── src/metric_engine/
│   ├── core/           # BaseMetric, Registry, Pipeline, Config
│   ├── io/             # InputSource, OutputManager, OpenAIClient
│   ├── metrics/        # keyword_search, pii_search, zero_shot, llm_classifier
│   ├── presets/safer/  # keywords.yaml, prompts.yaml, llm_prompts.yaml, pipeline.yaml
│   └── cli/            # CLI entrypoint
├── tests/              # 68 unit + integration tests
├── examples/           # demo.py, demo_notebook.ipynb, pipeline YAML
├── scripts/            # download_model.py
├── Dockerfile
├── pyproject.toml
└── requirements.txt
```
