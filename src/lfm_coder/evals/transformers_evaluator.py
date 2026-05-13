"""Evaluator for all transformers-based models.

This module provides the TransformersEvaluator implementation for evaluating
models using the Hugging Face transformers library.

Example usage:
    >>> from pprint import pprint
    >>> from lfm_coder.evals.transformers_evaluator import TransformersEvaluator
    >>> evaluator = TransformersEvaluator(
    ...     model_name="qwen3.5:0.8b",
    ...     model_id="qwen3.5-0.8b-q4_k_m-baseline",
    ... )
    >>> results = evaluator.evaluate(dataset_names=["human_eval", "mbpp"])
    >>> print(
    ...     f"HumanEval pass rate: {results.metrics['human_eval'].pass_rate:.1%}"
    ... )
    >>> print(f"MBPP pass rate: {results.metrics['mbpp'].pass_rate:.1%}")
    >>> pprint(results.metrics, indent=2)
"""

import time
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from lfm_coder.evals.eval import Evaluator
from lfm_coder.evals.types import GenerationResult
from lfm_coder.sandbox import SandboxType


class TransformersEvaluator(Evaluator):
    """Evaluator for models loaded via transformers.

    Attributes:
        model: The transformers model.
        tokenizer: The transformers tokenizer.
        device: Device to run generation on.
    """

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer
    device: str | torch.device

    def __init__(
        self,
        model_name: str,
        model_id: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        output_dir: str | Path = "data/evaluation/results",
        batch_size: int = 8,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        device: str | torch.device | None = None,
    ):
        if device is None:
            from lfm_coder.device import detect_device

            device = torch.device(detect_device())

        super().__init__(
            model_name=model_name,
            model_id=model_id,
            output_dir=output_dir,
            batch_size=batch_size,
            temperature=temperature,
            max_tokens=max_tokens,
            sandbox_type=SandboxType.DOCKER,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self, prompts: list[list[dict[str, str]]], **kwargs: Any
    ) -> list[GenerationResult]:
        """Generate completions using the transformers model."""

        # Format prompts using tokenizer's chat template
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                p, tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]

        # Tokenize prompts as a batch
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            # Use left padding for autoregressive generation
            padding_side="left",
        ).to(self.device)

        gen_kwargs = {
            "max_new_tokens": self.max_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)

        with torch.no_grad():
            start_time = time.perf_counter()
            # NOTE (Ryan Parker, 2026-04-24): ty has trouble identifying that `model`
            # is a PyTorch module, not a Tensor, so I ignore the type error for this line.
            outputs = self.model.generate(**inputs, **gen_kwargs)  # ty:ignore[call-non-callable]
            end_time = time.perf_counter()
            generation_time = end_time - start_time

        # Extract only the generated part (excluding the prompt)
        generated_ids = [
            output[len(inputs["input_ids"][i]) :] for i, output in enumerate(outputs)
        ]

        completions = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        results: list[GenerationResult] = []
        total_tokens = sum(len(item) for item in generated_ids)
        for i, completion in enumerate(completions):
            results.append(
                GenerationResult(
                    completion=completion,
                    token_count=len(generated_ids[i]),
                    # NOTE (Ryan Parker, 2026-04-24): This calculates the throughput
                    # over the entire batch. This is a bit misleading, but it's the best
                    # we can do since all samples are processed in a single batch.
                    generation_time=generation_time,
                    throughput=total_tokens / generation_time,
                )
            )

        return results
