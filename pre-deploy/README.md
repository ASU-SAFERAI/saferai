# Pre-Deploy Evaluation Metrics Library

A framework for comprehensive LLM evaluation metrics, providing reusable evaluation steps, perturbation strategies, and scoring pipelines.

## Installation

1. Make sure python version is >=3.11.

2. Create and activate a virtual env:

    ```bash
    python -m venv .venv && source .venv/bin/activate
    ```

3. Install `pre_deploy` package from git:

    ```bash
    pip install "git+ssh://git@github.com/<your-org>/<your-repo>.git@main"
    ```

4. Test import:
    ```python
    from pre_deploy import RequestDict

    request_dict = RequestDict(
        model_name="gpt4o_mini",
        model_provider="openai",
        username="your_username",
        metric_name="test_metric",
        run_id="your-run-id"
    )

    print(request_dict.to_dict())
    ```

## Structure

### `input`

This library uses its own input dataset format to keep its codebase agnostic of any existing evaluation frameworks. Datasets read in using `EvalDataset.from_dict(...)` should have the following dictionary format:

```python
from pre_deploy import EvalDataset

dataset = EvalDataset.from_dict({
    "id": str,  # Required attr
    "conversations": [  # Required attr
        {
            "id": str,  # Required attr
            "messages": [  # Required attr
                "sequence": int,  # Required attr; index of message in convo
                "role": str,  # Required attr; "user" or "assistant"
                "contents": [  # Required attr
                    "type": str,  # Required attr; image, video, audio, text_file, text, tool_call
                    "uri": str,
                    "content": str,
                    "metadata": {...}  # Insert your metadata here.
                ],
                "metadata": {...}  # Insert your metadata here.
            ],
            "metadata": {
                "expected_output": str  # Required only if using ground-truth eval metrics
            }  # Insert your metadata here.
        }
    ],
    "metadata": {...}  # Insert your metadata here.
})
```

**Some notes on conventions**:

1. Single-turn evaluations should consist of one conversation per exchange between the bot and user.
2. Multi-turn evaluations should consist of one conversation object per conversation between bot and assistant.

### Prompt Templates

#### `evaluation_steps`

These evaluation steps power streamlined custom GEvals from DeepEval. They are a list of yes/no steps provided to the GEval interface that provide easy-to-use scores for your evaluation use cases.

Available templates:

1. Direct Answering Avoidance score (useful for educational chatbots).
2. Socratic score (useful for tutor bots that follow the Socratic method).
3. Tone (useful for any stakeholder-facing bot).
4. Accuracy (with ground truth, used in other modules of this library).

```python
from pre_deploy.evaluation_steps import (
    DIRECT_ANSWERING_EVAL_STEPS,
    SOCRATIC_EVAL_STEPS,
    TONE_EVAL_STEPS
)
```

#### `deep_acylic_graphs`

These DAGs provide deterministic evaluations with an if/then decision-tree style interface.

```python
from pre_deploy.deep_acylic_graphs import FACT_CHECKER_DAG
```

### `query_processor`

The query processor sends LLM queries to a batch processing backend (SQS + DynamoDB). It queues all queries and processes them in batches, which is faster than sequential API calls.

```python
from pre_deploy.query_processor import DeepEvalClient, RequestDict

model = DeepEvalClient(
    RequestDict(
        username="[USERNAME]",
        metric_name="[METRIC_NAME]",
        model_name="[MODEL_NAME]",
        model_provider="[MODEL_PROVIDER]",
        run_id="[RUN_ID]"
    )
)
```

### `metrics`

Metrics take an `EvalDataset` and score it directly. Available metrics:

| Metric | `metric_name` |
|--------|---------------|
| Accuracy | `Accuracy` |
| BBQ | `bias` |
| BoolQ | `reading_comprehension` |
| CARE | `CARE` |
| Consistency | `Consistency` |
| Conversation Completeness | `Conversation Completeness` |
| Conversation GEval | `Conversation GEval` |
| Conversation Relevancy | `Conversation Relevancy` |
| GEval | `GEval` |
| MARBLE | `MARBLE` |
| Robustness | `Robustness` |
| Tool Call Failure Modes | `tool_call_failure_modes` |

#### BBQ (Bias Benchmark for Question-Answering)

```python
from pre_deploy.metrics.bbq import BBQ
from pre_deploy.query_processor import DeepEvalClient, RequestDict

model = DeepEvalClient(RequestDict(
    username="your_username",
    metric_name="bias",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
))

bbq = BBQ(bias_type="Age", model=model)

accuracy_score = bbq.score
refusal_rate = bbq.refusal_rate
amb_bias_score = bbq.amb_bias_score
disamb_bias_score = bbq.disamb_bias_score
```

#### BoolQ (Boolean Questions)

```python
from pre_deploy.metrics.boolq import BoolQ
from pre_deploy.query_processor import DeepEvalClient, RequestDict

model = DeepEvalClient(RequestDict(
    username="your_username",
    metric_name="reading_comprehension",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
))

boolq = BoolQ(model=model)

accuracy = boolq.score
refusal_rate = boolq.refusal_rate
detailed_stats = boolq.stats
```

#### CARE (Context-Aware Responsible Evaluation)

```python
from pre_deploy.metrics.care import CARE
from pre_deploy.query_processor import DeepEvalClient, RequestDict

model = DeepEvalClient(RequestDict(
    username="your_username",
    metric_name="CARE",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
))

care = CARE(model=model, threshold=0.5)
results = care.results
results_dict = results.to_dict()
```

#### MARBLE (Multi-dimensional Assessment of Reasoning and Bias)

```python
from pre_deploy.metrics.marble import MARBLE
from pre_deploy.query_processor import DeepEvalClient, RequestDict

model = DeepEvalClient(RequestDict(
    username="your_username",
    metric_name="MARBLE",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
))

marble = MARBLE(model=model, threshold=0.5)
results = marble.results
results_dict = results.to_dict()
```

#### Consistency

```python
from pre_deploy.metrics.consistency import consistency_batch_generate
from pre_deploy.query_processor import RequestDict
from pre_deploy import EvalDataset

evaluator_info = RequestDict(
    username="your_username",
    metric_name="Consistency",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
)

eval_dataset = EvalDataset.from_dict({...})
results = consistency_batch_generate(
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    threshold=0.5
)
```

#### GEval

```python
from pre_deploy.metrics.geval import geval_batch_generate
from pre_deploy.query_processor import RequestDict
from pre_deploy import EvalDataset

evaluator_info = RequestDict(
    username="your_username",
    metric_name="GEval",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
)

eval_dataset = EvalDataset.from_dict({...})
evaluation_steps = ["Step 1: Check if response is helpful", "Step 2: Check if response is accurate"]

results = geval_batch_generate(
    name="geval evaluation",
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    with_reference=False,
    evaluation_steps=evaluation_steps,
    threshold=0.5
)
```

#### DAG (Deep Acyclic Graph)

```python
from pre_deploy.metrics.dag import dag_batch_generate
from pre_deploy.deep_acylic_graphs import FACT_CHECKER_DAG
from pre_deploy.query_processor import RequestDict
from pre_deploy import EvalDataset

evaluator_info = RequestDict(
    username="your_username",
    metric_name="dag_metric",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id"
)

eval_dataset = EvalDataset.from_dict({...})
results = dag_batch_generate(
    name="fact checker evaluation",
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    dag=FACT_CHECKER_DAG,
    threshold=0.5
)
```

#### Tool Call Failure Modes

Scores an ordered tool-calling trace against a golden trace. Unlike the G-Eval
metrics, the judge returns a *structured* verdict: an `overall_score` (0-1,
alignment with the golden truth) plus six binary quality dimensions where 1 is
good and 0 is bad:

| Dimension | Meaning |
|-----------|---------|
| `tool_correctness` | Ordered selected tool sequence matches the golden trace (or correctly abstains). |
| `tool_potentially_valid` | Tool choices are plausible for the request, even if they differ from the golden sequence. |
| `json_syntax` | Response was syntactically valid JSON only — no prose, fences, or surrounding text. |
| `required_args_present` | Every schema-required argument for each chosen tool is present (presence only). |
| `optional_args_present` | Non-default optional args from the golden truth are present and matched, with no efficiency-harming extras. |
| `placeholder_propagation` | Each `{{steps[N].output}}` dependency in the golden trace is satisfied by the consuming call's observed output. |

The metric follows the same staged lifecycle as the other asynchronous metrics.
Use the single-shot convenience function to enqueue, wait, and score in one call:

```python
from pre_deploy.metrics.tool_call_failure_modes import (
    tool_call_failure_modes_batch_generate,
)
from pre_deploy.query_processor import RequestDict
from pre_deploy import EvalDataset

evaluator_info = RequestDict(
    username="your_username",
    metric_name="tool_call_failure_modes",
    model_name="gpt4o_mini",
    model_provider="openai",
    run_id="your-run-id",
)

eval_dataset = EvalDataset.from_dict({...})
results = tool_call_failure_modes_batch_generate(
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    threshold=0.5,
)
```

Or drive the two-phase lifecycle directly — `start_...` enqueues the judge
prompts and returns immediately, and `finalize_...` polls, returning a status
dict (`is_complete=False`) until every item is scored, then a `MetricsResults`:

```python
from pre_deploy.metrics.tool_call_failure_modes import (
    start_tool_call_failure_modes,
    finalize_tool_call_failure_modes,
)

start_tool_call_failure_modes(
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    threshold=0.5,
)

result = finalize_tool_call_failure_modes(
    evaluator_info=evaluator_info,
    eval_dataset=eval_dataset,
    threshold=0.5,
)
```

Each conversation's `system_prompt` metadata supplies the tool catalog and output
rules the judge compares against. The per-scenario verdict is carried on the
returned `MetricsResults`, and the six dimensions are exposed through its
`to_dict()`.

A sample dataset is bundled at
`pre_deploy/data/tool_calling/tool_calling_seed_60_populated_multihop.json` for
use as example input.

### Output Format

All metrics (except BBQ and BoolQ) yield a `MetricsResults` object:

```python
{
    "results": {
        "convo_id1": {
            "score": 0.9,
            "reason": "Did a good job overall.",
            "success": True  # threshold set by user
        }
    }
}
```
