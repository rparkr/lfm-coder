"""OpenAI-compatible evaluator.

This module provides the OpenAICompatibleEvaluator implementation for evaluating
models via OpenAI-compatible APIs (e.g., Ollama, vLLM).

Example usage:
    >>> from pprint import pprint
    >>> from lfm_coder.evals.openai_evaluator import OpenAICompatibleEvaluator
    >>> evaluator = OpenAICompatibleEvaluator(
    ...     model_name="qwen3.5:0.8b",
    ...     model_id="qwen3.5-0.8b-q4_k_m-baseline",
    ...     base_url="http://localhost:11434/v1",
    ...     api_key="ollama",
    ... )
    >>> results = evaluator.evaluate(dataset_names=["human_eval", "mbpp"])
    >>> print(f"HumanEval pass rate: {results.metrics['human_eval'].pass_rate:.1%}")
    >>> print(f"MBPP pass rate: {results.metrics['mbpp'].pass_rate:.1%}")
    >>> pprint(results.metrics, indent=2)
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx

from lfm_coder.evals.eval import Evaluator
from lfm_coder.evals.types import GenerationResult
from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox import SandboxType

logger = get_logger(__name__)


class OpenAICompatibleEvaluator(Evaluator):
    """Evaluator for OpenAI-compatible API endpoints (e.g., Ollama, vLLM).

    Attributes:
        base_url: Base URL for the API.
        api_key: API key for authentication.
        model_name: Name of the model to evaluate.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        base_url: str | None = None,
        api_key: str | None = None,
        output_dir: str | Path = "data/evaluation/results",
        batch_size: int = 16,
        temperature: float = 0.0,
        max_tokens: int | None = 16_384,
        sandbox_type: SandboxType = SandboxType.AUTO,
    ):
        super().__init__(
            model_name=model_name,
            model_id=model_id,
            output_dir=output_dir,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            sandbox_type=sandbox_type,
        )
        self.base_url = base_url or os.getenv(
            "OPENAI_BASE_URL", "http://localhost:11434/v1"
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "ollama")

    def generate(
        self, prompts: list[list[dict[str, str]]], **kwargs: Any
    ) -> list[GenerationResult]:
        """Generate completions using the OpenAI-compatible API."""
        return asyncio.run(self._generate_async(prompts, **kwargs))

    async def _generate_async(
        self, prompts: list[list[dict[str, str]]], **kwargs: Any
    ) -> list[GenerationResult]:
        """Generate completions concurrently."""
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=120.0,
        ) as client:
            tasks = [
                self._generate_single(client, prompt, **kwargs) for prompt in prompts
            ]
            return await asyncio.gather(*tasks)

    async def _generate_single(
        self, client: httpx.AsyncClient, prompt: list[dict[str, str]], **kwargs: Any
    ) -> GenerationResult:
        """Generate a single completion and track metrics."""
        try:
            payload = {
                "model": self.model_name,
                "messages": prompt,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
                # Disable thinking for reasoning models. This codebase focuses on
                # improving code generation capabilities for instruct-tuned models
                # that have fast response times compared to reasoning models.
                "chat_template_kwargs": {"enable_thinking": False},
            }
            payload.update(kwargs)

            start_time = time.perf_counter()
            response = await client.post("/chat/completions", json=payload)
            end_time = time.perf_counter()
            duration = end_time - start_time

            response.raise_for_status()

            data = response.json()
            completion = data["choices"][0]["message"]["content"]
            token_count = data.get("usage", {}).get("completion_tokens", 0)

            throughput = token_count / duration if duration > 0 else 0

            return GenerationResult(
                completion=completion,
                token_count=token_count,
                generation_time=duration,
                throughput=throughput,
            )
        except Exception as e:
            logger.error(
                f"[{self.model_id}] API call failed during evaluation. Error: {e}"
            )
            return GenerationResult(
                completion="",
                token_count=0,
                generation_time=0,
                throughput=0,
            )
