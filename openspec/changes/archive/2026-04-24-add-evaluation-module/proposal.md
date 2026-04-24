## Why

The lfm-coder project trains models using GRPO with RLVR via TRL, but lacks a way to measure model performance during training. We need an evaluation module to track progress on HumanEvalPlus and MBPPPlus benchmarks at regular intervals, enabling data-driven decisions about training duration and model quality.

## What Changes

- Create a new `eval.py` module in `src/lfm_coder/evals/` that evaluates transformers models on evaluation datasets
- Add a separate `OpenAICompatibleEvaluator` class for evaluating models via OpenAI-compatible APIs (e.g., Ollama)
- Implement configurable evaluation frequency (default: every 100 steps)
- Add support for temperature and batch size configuration
- Track comprehensive metrics: pass rate, code extraction success, format success, token counts, execution time, per-task accuracy
- Implement progress bars with tqdm
- Add logging throughout for debugging
- Use trackio for metrics and save results to JSONL files
- Implement resume capability from last checkpoint

## Capabilities

### New Capabilities

- `transformers-evaluation`: Evaluate transformers models on HumanEvalPlus and MBPPPlus benchmarks with configurable parameters and comprehensive metrics
- `openai-compatible-evaluation`: Evaluate any OpenAI-compatible API endpoint (e.g., Ollama) using the same evaluation logic
- `evaluation-checkpointing`: Save and resume evaluation progress to enable recovery from failures

## Impact

- New file: `src/lfm_coder/evals/eval.py` - Main evaluation module
- New file: `src/lfm_coder/evals/openai_evaluator.py` - OpenAI-compatible evaluator
- Dependencies: `transformers`, `trackio`, `tqdm`, `openai` (or `httpx`)
- Integration point: Training loop will call evaluation at configurable intervals