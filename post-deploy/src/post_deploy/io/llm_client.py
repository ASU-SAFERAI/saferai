"""Generic LLM client adapter for calling OpenAI-compatible APIs."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    Abstract interface for LLM API clients.

    Subclass this to integrate with different LLM providers.
    """

    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Send a single prompt to the LLM and return the response text.

        Args:
            prompt: The full prompt string to send.

        Returns:
            The model's response as a string.
        """
        ...

    def batch_query(self, prompts: Dict[int, str], max_concurrent: int = 4) -> Dict[int, str]:
        """
        Send multiple prompts and return responses keyed by the same IDs.

        Default implementation calls query() sequentially.
        Subclasses can override for parallel/batch execution.

        Args:
            prompts: Dict mapping row index -> prompt string.
            max_concurrent: Max concurrent requests (for async implementations).

        Returns:
            Dict mapping row index -> response string.
        """
        results = {}
        total = len(prompts)

        for i, (idx, prompt) in enumerate(prompts.items()):
            logger.debug("Querying LLM [%d/%d]", i + 1, total)
            try:
                results[idx] = self.query(prompt)
            except Exception as e:
                logger.warning("LLM query failed for index %d: %s", idx, e)
                results[idx] = ""

        return results


class OpenAIClient(BaseLLMClient):
    """
    LLM client using the OpenAI Python SDK.

    Works with any OpenAI-compatible API endpoint including:
    - OpenAI (api.openai.com)
    - Azure OpenAI
    - vLLM / TGI served models
    - Ollama (localhost)
    - Any server implementing the /v1/chat/completions spec

    Requires: pip install openai

    Args:
        api_key: API key for authentication.
        base_url: Base URL of the API (default: https://api.openai.com/v1).
            For local servers: "http://localhost:11434/v1"
            For Azure: "https://{resource}.openai.azure.com/openai/deployments/{deployment}"
        model: Model name (e.g., "gpt-4", "gpt-3.5-turbo", "llama3").
        temperature: Sampling temperature (default: 0.0 for deterministic classification).
        max_tokens: Maximum tokens in the response (default: 128, sufficient for labels).
        timeout: Request timeout in seconds (default: 60).
        system_prompt: Optional system message prepended to every request.
        extra_params: Additional parameters passed to the API (top_p, frequency_penalty, etc.).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 60,
        system_prompt: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.extra_params = extra_params or {}
        self._client = None  # Lazy-loaded

    def _get_client(self):
        """Lazy-load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "The 'openai' package is required for OpenAIClient. "
                    "Install with: pip install openai"
                )
            self._client = OpenAI(
                api_key=self.api_key or "no-key-required",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    def query(self, prompt: str) -> str:
        """
        Send a prompt to the OpenAI-compatible chat completions endpoint.

        Args:
            prompt: The user message content.

        Returns:
            The assistant's response text.
        """
        client = self._get_client()

        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **self.extra_params,
        )

        return response.choices[0].message.content.strip()

    def batch_query(self, prompts: Dict[int, str], max_concurrent: int = 4) -> Dict[int, str]:
        """
        Send multiple prompts concurrently using a thread pool.

        Args:
            prompts: Dict mapping row index -> prompt string.
            max_concurrent: Max concurrent threads.

        Returns:
            Dict mapping row index -> response string.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: Dict[int, str] = {}

        def _query_one(item):
            idx, prompt = item
            try:
                return idx, self.query(prompt)
            except Exception as e:
                logger.warning("LLM query failed for index %d: %s", idx, e)
                return idx, ""

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = {
                executor.submit(_query_one, (idx, prompt)): idx
                for idx, prompt in prompts.items()
            }
            for future in as_completed(futures):
                idx, response = future.result()
                results[idx] = response

        return results


# Keep backward compatibility alias
HTTPLLMClient = OpenAIClient
