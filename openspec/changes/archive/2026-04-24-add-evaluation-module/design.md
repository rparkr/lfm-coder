## Context

The lfm-coder project trains models using GRPO with RLVR via TRL. Currently, there's no way to measure model performance during training. The project has:
- Two evaluation datasets: HumanEvalPlus and MBPPPlus (in `src/lfm_coder/datasets/eval_data.py`)
- A unified Sandbox for code execution (in `src/lfm_coder/sandbox/sandbox.py`)
- Reward helpers for pass rate calculation (in `src/lfm_coder/rewards/helpers.py`)

The evaluation module needs to:
1. Generate code completions from a model
2. Execute the generated code in the sandbox with test cases
3. Calculate pass rates and other metrics
4. Track progress and enable recovery from failures

## Goals / Non-Goals

**Goals:**
- Create an evaluation module that measures model performance on HumanEvalPlus and MBPPPlus
- Support both transformers models and OpenAI-compatible API endpoints
- Track comprehensive metrics: pass rate, code extraction success, format success, token counts, execution time, per-task accuracy
- Enable configurable evaluation frequency (default: every 100 steps)
- Implement progress bars with tqdm
- Add logging throughout for debugging
- Use trackio for metrics and save results to JSONL files
- Implement resume capability from last checkpoint

**Non-Goals:**
- Implement the actual training loop integration (that's a separate change)
- Support evaluation on training data
- Implement model-specific optimizations for generation

## Decisions

### 1. Code Extraction Strategy

**Decision:** Extract code from the last fenced code block in the model's output.

**Rationale:** Models often output markdown with code blocks. The last block is typically the final answer. If there's an opening fence but no closing one, extract from the start of the code block to the end of output.

**Alternatives considered:**
- Extract from first code block: Less reliable, models often include reasoning before final code
- Use regex to find any code block: Could match intermediate code that isn't the final answer

### 2. Two Evaluator Classes

**Decision:** Create a base `Evaluator` class with shared logic, then extend with `TransformersEvaluator` and `OpenAICompatibleEvaluator`.

**Rationale:** This maximizes code reuse while allowing different generation strategies. The base class handles dataset iteration, metric calculation, checkpointing, and logging. Subclasses only implement the `generate` method.

**Alternatives considered:**
- Single class with conditional logic: Would lead to complex if/else branches
- Protocol-based approach: Over-engineered for this use case

### 3. Checkpoint Strategy

**Decision:** Save evaluation results to JSONL after each batch, and resume by reading the last completed batch from the file.

**Rationale:** Simple and robust. The JSONL file serves as both a log and a checkpoint. On resume, we read the file, find the last completed task_id, and continue from there.

**Alternatives considered:**
- Separate checkpoint file: Extra complexity, potential for inconsistency
- In-memory only: No recovery capability

### 4. MBPPPlus Test Selection

**Decision:** Use only the "long" tests for MBPPPlus evaluation.

**Rationale:** The "long" tests are more comprehensive and are the standard benchmark. The "short" tests are simpler and may not catch as many edge cases.

**Alternatives considered:**
- Use both: Would double evaluation time without significant benefit
- Use short tests only: Less comprehensive evaluation

## Risks / Trade-offs

- **[Risk]** Model generation could be slow, especially for large models
  - **Mitigation:** Use batch processing and configurable batch size

- **[Risk]** Code extraction could fail for malformed model outputs
  - **Mitigation:** Track format success rate as a metric, log failures for debugging

- **[Risk]** Sandbox execution could hang or timeout
  - **Mitigation:** Use configurable timeouts, track execution time per sample

- **[Risk]** Evaluation could take a long time for large datasets
  - **Mitigation:** Implement resume capability, use tqdm for progress tracking

- **[Risk]** JSONL file could grow large
  - **Mitigation:** Each evaluation run creates a new file with timestamp

## Open Questions

- Should we support multiple temperature values in a single evaluation run?
- Should we track additional metrics like latency percentiles?
- Should we integrate with Weights & Biases or TensorBoard directly, or just use trackio?