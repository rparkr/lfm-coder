"""Evaluate LLMs on coding benchmarks.

Supports the following evaluators:

- TransformersEvaluator: For models loaded via the transformers library.
- OpenAICompatibleEvaluator: For models with OpenAI-compatible APIs (e.g., Ollama, vLLM).

Supported datasets:
- human_eval: HumanEval benchmark.
- mbpp: MBPP benchmark.
"""

from lfm_coder.evals.openai_evaluator import OpenAICompatibleEvaluator
from lfm_coder.evals.transformers_evaluator import TransformersEvaluator

__all__ = [
    "OpenAICompatibleEvaluator",
    "TransformersEvaluator",
]
