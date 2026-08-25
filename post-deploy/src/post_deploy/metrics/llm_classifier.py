"""LLM-based classifier metric: classify text by sending prompts to an external LLM API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from post_deploy.core.metric import BaseMetric, MetricContext
from post_deploy.io.llm_client import BaseLLMClient, OpenAIClient

logger = logging.getLogger(__name__)


class LLMClassifierMetric(BaseMetric):
    """
    Classifies text columns by sending structured prompts to an external LLM API.

    The LLM returns comma-separated labels which are parsed, validated against
    expected categories, and converted to boolean columns (one per category).

    Configuration options:
        target_columns: list[str] - logical column names to classify (e.g., ["query", "response"])
        prompts: list[dict] - inline prompt definitions, each with:
            - template: str - prompt template with {text} placeholder for the input
            - categories: list[str] - expected output categories
            - target: str - which logical column to classify ("query" or "response")
            - prefix: str - prefix for output column names (e.g., "llm_query")
        prompts_file: str - path to a YAML file with prompt definitions
        preset: str - name of a preset to load prompts from (e.g., "safer")
        llm_client: dict - OpenAI-compatible client configuration:
            - base_url: str - API base URL (e.g., "https://api.openai.com/v1")
            - api_key: str - authentication key/token
            - model: str - model name (default: "gpt-4")
            - temperature: float - sampling temperature (default: 0.0)
            - max_tokens: int - max response tokens (default: 128)
            - max_concurrent: int - max concurrent requests (default: 4)
            - timeout: int - request timeout in seconds (default: 60)
            - system_prompt: str - optional system message
            - extra_params: dict - additional API parameters
        client_instance: BaseLLMClient - pre-built client (for programmatic use / testing)
    """

    NAME = "llm_classifier"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._target_columns: List[str] = self._config.get("target_columns", ["query", "response"])
        self._prompts: List[Dict[str, Any]] = []
        self._client: Optional[BaseLLMClient] = self._config.get("client_instance")
        self._max_concurrent: int = self._config.get("llm_client", {}).get("max_concurrent", 4)

        self._load_prompts()

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required_columns(self) -> List[str]:
        return self._target_columns

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Validate LLM classifier configuration."""
        target_columns = config.get("target_columns", ["query", "response"])
        if not target_columns:
            raise ValueError("LLMClassifierMetric requires at least one 'target_columns' entry.")

        has_prompts = config.get("prompts") or config.get("prompts_file") or config.get("preset")
        if not has_prompts:
            raise ValueError(
                "LLMClassifierMetric requires at least one prompt source: "
                "'prompts' (inline list), 'prompts_file' (YAML path), or 'preset' (preset name)."
            )

        # Client config or instance is needed
        has_client = config.get("llm_client") or config.get("client_instance")
        if not has_client:
            raise ValueError(
                "LLMClassifierMetric requires 'llm_client' config (with api_url) "
                "or a pre-built 'client_instance'."
            )

    def process(self, df: pd.DataFrame, context: MetricContext) -> pd.DataFrame:
        """Run LLM classification on configured target columns."""
        self._ensure_client()

        for prompt_def in self._prompts:
            target = prompt_def["target"]
            actual_col = context.get_column(target)
            df = self._classify_with_prompt(df, actual_col, prompt_def)

        return df

    def _classify_with_prompt(
        self, df: pd.DataFrame, actual_col: str, prompt_def: Dict[str, Any]
    ) -> pd.DataFrame:
        """Classify a column using a single prompt definition."""
        template = prompt_def["template"]
        categories = prompt_def["categories"]
        prefix = prompt_def.get("prefix", f"llm_{prompt_def['target']}")

        logger.info(
            "LLM classifying column '%s' (%d rows, %d categories, prefix='%s')",
            actual_col, len(df), len(categories), prefix,
        )

        # Build prompts for each row
        batch_prompts: Dict[int, str] = {}
        for idx, text in df[actual_col].items():
            batch_prompts[idx] = template.format(text=str(text))

        # Send to LLM
        responses = self._client.batch_query(batch_prompts, max_concurrent=self._max_concurrent)

        # Parse responses into labels
        parsed_labels = self._parse_responses(responses, categories)

        # Create boolean columns for each category
        for category in categories:
            col_name = f"{prefix}_{category}"
            df[col_name] = df.index.map(
                lambda idx, cat=category: cat in parsed_labels.get(idx, [])
            )

        # Also store the raw LLM response for debugging
        raw_col = f"{prefix}_raw_response"
        df[raw_col] = df.index.map(lambda idx: responses.get(idx, ""))

        return df

    @staticmethod
    def _parse_responses(
        responses: Dict[int, str], expected_categories: List[str]
    ) -> Dict[int, List[str]]:
        """
        Parse comma-separated LLM responses into validated label lists.

        Args:
            responses: Dict mapping row index -> raw response string.
            expected_categories: List of valid category names.

        Returns:
            Dict mapping row index -> list of validated labels.
        """
        parsed: Dict[int, List[str]] = {}

        for idx, raw_response in responses.items():
            # Split by comma, strip whitespace, lowercase
            raw_labels = [label.strip().lower().replace(" ", "_") for label in raw_response.split(",")]

            # Validate against expected categories
            valid_labels = [label for label in raw_labels if label in expected_categories]

            if not valid_labels and raw_labels:
                logger.debug(
                    "Row %d: no valid labels found in response '%s'. Expected: %s",
                    idx, raw_response, expected_categories,
                )

            parsed[idx] = valid_labels

        return parsed

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
    def _load_prompts_from_yaml(path: str) -> List[Dict[str, Any]]:
        """Load prompts from a YAML file."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"LLM prompts file not found: {filepath}")

        with open(filepath) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, list):
            raise ValueError(
                f"LLM prompts YAML must be a list of prompt definitions, got {type(data).__name__}"
            )

        # Validate each prompt
        for i, prompt in enumerate(data):
            if "template" not in prompt:
                raise ValueError(f"LLM prompt #{i} missing 'template'.")
            if "categories" not in prompt:
                raise ValueError(f"LLM prompt #{i} missing 'categories'.")
            if "target" not in prompt:
                raise ValueError(f"LLM prompt #{i} missing 'target' (logical column name).")
            if "{text}" not in prompt["template"]:
                raise ValueError(
                    f"LLM prompt #{i} template must contain '{{text}}' placeholder."
                )

        return data

    @staticmethod
    def _load_prompts_from_preset(preset_name: str) -> List[Dict[str, Any]]:
        """Load prompts from a named preset."""
        from post_deploy.presets.safer import PRESET_DIR

        if preset_name == "safer":
            prompts_path = PRESET_DIR / "llm_prompts.yaml"
            if not prompts_path.exists():
                raise FileNotFoundError(
                    f"SAFER preset LLM prompts file not found at {prompts_path}. "
                    "Ensure the preset files are installed."
                )
            with open(prompts_path) as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(f"Unknown preset: '{preset_name}'. Available presets: ['safer']")

    def _ensure_client(self) -> None:
        """Build the LLM client from config if not already set."""
        if self._client is not None:
            return

        client_config = self._config.get("llm_client", {})
        api_key = client_config.get("api_key", "")
        base_url = client_config.get("base_url", client_config.get("api_url", ""))

        if not base_url:
            raise ValueError(
                "LLMClassifierMetric requires 'llm_client.base_url' in config "
                "or a pre-built 'client_instance'."
            )

        self._client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            model=client_config.get("model", "gpt-4"),
            temperature=client_config.get("temperature", 0.0),
            max_tokens=client_config.get("max_tokens", 128),
            timeout=client_config.get("timeout", 60),
            system_prompt=client_config.get("system_prompt"),
            extra_params=client_config.get("extra_params"),
        )
        self._max_concurrent = client_config.get("max_concurrent", 4)

        logger.info(
            "Initialized OpenAIClient (model=%s, base_url=%s)",
            self._client.model, self._client.base_url,
        )
