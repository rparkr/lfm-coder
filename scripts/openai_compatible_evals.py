"""
Run evals using OpenAI-compatible models.

This is a quick demo to test the evaluation functionality.

Usage:
    uv run scripts/openai_compatible_evals.py
"""

from pprint import pprint

from lfm_coder.evals.openai_evaluator import OpenAICompatibleEvaluator

MODEL_NAME = "lfm2.5-instruct"
# MODEL_NAME = "lfm2.5-thinking"
# MODEL_NAME = "qwen3.5:4b"
# MODEL_NAME = "qwen3:0.6b"
# MODEL_NAME = "qwen3.5:0.8b"
MODEL_ID = f"{MODEL_NAME}-q4_k_m-baseline"

evaluator = OpenAICompatibleEvaluator(
    model_name=MODEL_NAME,
    model_id=MODEL_ID,
    base_url="http://localhost:11434/v1",
    batch_size=16,
    temperature=0.0,
    max_tokens=16_384,
)
results = evaluator.evaluate(dataset_names=["human_eval", "mbpp"], resume=True)
print(f"HumanEval Pass@1: {results.metrics['human_eval'].pass_rate:.1%}")
print(f"MBPP Pass@1: {results.metrics['mbpp'].pass_rate:.1%}")
pprint(results.metrics, indent=2)
