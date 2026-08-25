"""Zero-shot classification metric: classify text using HuggingFace zero-shot models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from post_deploy.core.metric import BaseMetric, MetricContext

logger = logging.getLogger(__name__)


class ZeroShotMetric(BaseMetric):
    """
    Classifies text columns using zero-shot classification via HuggingFace Transformers.

    Requires the optional ML dependencies:
        pip install post-deploy[ml]

    Configuration options:
        target_columns: list[str] - logical column names to classify (e.g., ["query", "response"])
        prompts: list[dict] - inline prompt definitions, each with:
            - hypothesis_template: str - template with {} placeholder
            - labels: dict[str, str] - mapping of output_col_name -> label_text
            - target: str - which logical column to apply to ("query" or "response")
        prompts_file: str - path to a YAML file with prompt definitions
        preset: str - name of a preset to load prompts from (e.g., "safer")
        model: str - model name or local path (default: "./process/zeroshot_offline")
        batch_size: int - batch size for inference (default: 8 with GPU, 1 without)
        use_gpu: bool | None - force GPU on/off. None = auto-detect (default: None)
        multi_label: bool - whether to use multi-label classification (default: True)
    """

    NAME = "zero_shot"

    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config or {}
        self._target_columns: list[str] = self._config.get("target_columns", ["query", "response"])
        self._model_path: str = self._config.get("model", "./process/zeroshot_offline")
        self._use_gpu: bool | None = self._config.get("use_gpu", None)
        self._batch_size: int | None = self._config.get("batch_size", None)
        self._multi_label: bool = self._config.get("multi_label", True)
        self._prompts: list[dict[str, Any]] = []
        self._classifier = None  # Lazy-loaded

        self._load_prompts()

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required_columns(self) -> list[str]:
        return self._target_columns

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate zero-shot classification configuration."""
        target_columns = config.get("target_columns", ["query", "response"])
        if not target_columns:
            raise ValueError("ZeroShotMetric requires at least one 'target_columns' entry.")

        has_prompts = config.get("prompts") or config.get("prompts_file") or config.get("preset")
        if not has_prompts:
            raise ValueError(
                "ZeroShotMetric requires at least one prompt source: "
                "'prompts' (inline list), 'prompts_file' (YAML path), or 'preset' (preset name)."
            )

        # Validate ML dependencies
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import datasets  # noqa: F401
        except ImportError:
            raise ValueError(
                "ZeroShotMetric requires 'transformers', 'torch', and 'datasets'. "
                "Install them with: pip install post-deploy[ml]"
            )

    def process(self, df: pd.DataFrame, context: MetricContext) -> pd.DataFrame:
        """Run zero-shot classification on configured columns."""
        self._ensure_classifier()

        for prompt_def in self._prompts:
            target = prompt_def["target"]
            actual_col = context.get_column(target)
            df = self._classify_with_prompt(df, actual_col, prompt_def)

        return df

    def _classify_with_prompt(
        self, df: pd.DataFrame, actual_col: str, prompt_def: dict[str, Any]
    ) -> pd.DataFrame:
        """Classify a column using a single prompt definition."""
        from datasets import Dataset

        hypothesis_template = prompt_def["hypothesis_template"]
        labels = prompt_def["labels"]  # {output_col_name: label_text}
        label_values = list(labels.values())
        col_name_map = {v: k for k, v in labels.items()}  # label_text -> output_col_name

        logger.info(
            "Classifying column '%s' with template '%s' (%d labels)",
            actual_col, hypothesis_template, len(label_values),
        )

        dataset = Dataset.from_pandas(df[[actual_col]].copy())

        def classify_batch(batch):
            results = self._classifier(
                batch[actual_col],
                candidate_labels=label_values,
                hypothesis_template=hypothesis_template,
                batch_size=self._effective_batch_size,
                multi_label=self._multi_label,
            )

            output = {label: [] for label in label_values}
            for res in results:
                for label, score in zip(res["labels"], res["scores"]):
                    output[label].append(score)
            return output

        classified = dataset.map(classify_batch, batched=True, batch_size=self._effective_batch_size)
        result_df = classified.to_pandas()

        # Rename label columns to the configured output column names
        result_df = result_df.rename(columns=col_name_map)

        # Only take the score columns (not the original text column)
        score_cols = list(labels.keys())
        for col in score_cols:
            df[col] = result_df[col].values

        return df

    def _load_prompts(self) -> None:
        """Load prompt definitions from config sources."""
        self._prompts = []

        # 1. Inline prompts
        inline = self._config.get("prompts")
        if inline:
            self._prompts.extend(inline)

        # 2. From YAML file
        prompts_file = self._config.get("prompts_file")
        if prompts_file:
            file_prompts = self._load_prompts_from_yaml(prompts_file)
            self._prompts.extend(file_prompts)

        # 3. From preset
        preset_name = self._config.get("preset")
        if preset_name:
            preset_prompts = self._load_prompts_from_preset(preset_name)
            self._prompts.extend(preset_prompts)

    @staticmethod
    def _load_prompts_from_yaml(path: str) -> list[dict[str, Any]]:
        """Load prompts from a YAML file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Prompts file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError(f"Prompts YAML must be a list of prompt definitions, got {type(data).__name__}")

        # Validate each prompt
        for i, prompt in enumerate(data):
            if "hypothesis_template" not in prompt:
                raise ValueError(f"Prompt #{i} missing 'hypothesis_template'.")
            if "labels" not in prompt:
                raise ValueError(f"Prompt #{i} missing 'labels'.")
            if "target" not in prompt:
                raise ValueError(f"Prompt #{i} missing 'target' (logical column name).")

        return data

    @staticmethod
    def _load_prompts_from_preset(preset_name: str) -> list[dict[str, Any]]:
        """Load prompts from a named preset."""
        from post_deploy.presets import safer as safer_preset

        if preset_name == "safer":
            prompts_path = safer_preset.PRESET_DIR / "prompts.yaml"
            if not prompts_path.exists():
                raise FileNotFoundError(
                    f"SAFER preset prompts file not found at {prompts_path}. "
                    "Ensure the preset files are installed."
                )
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown preset: '{preset_name}'. Available presets: ['safer']")

    def _ensure_classifier(self) -> None:
        """Lazy-load the HuggingFace zero-shot classification pipeline."""
        if self._classifier is not None:
            return

        try:
            import torch
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError:
            raise RuntimeError(
                "transformers and torch are not installed. "
                "Install them with: pip install post-deploy[ml]"
            )

        gpu_available = torch.cuda.is_available()
        use_gpu = self._use_gpu if self._use_gpu is not None else gpu_available

        self._effective_batch_size = self._batch_size or (8 if use_gpu else 1)

        logger.info(
            "Initializing zero-shot classifier (model=%s, gpu=%s, batch_size=%d)",
            self._model_path, use_gpu, self._effective_batch_size,
        )

        model_path = Path(self._model_path)

        if model_path.is_dir():
            # Local model
            logger.info("Loading model from local directory: %s", model_path)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True
            )
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        else:
            # HuggingFace Hub model
            logger.info("Loading model from HuggingFace Hub: %s", self._model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_path)
            tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        self._classifier = pipeline(
            task="zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if use_gpu else "cpu",
        )
