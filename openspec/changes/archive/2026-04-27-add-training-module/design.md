## Context

The `lfm-coder` project is designed to improve the coding capabilities of Liquid AI's LFM models. It already possesses a robust execution engine (Monty/Docker sandboxes) and dataset processing utilities. The missing piece is the actual training loop that ties these together using Group Relative Policy Optimization (GRPO). The primary hardware constraint is a single NVIDIA RTX 4060 with 8GB of VRAM.

## Goals / Non-Goals

**Goals:**
- Implement a stable GRPO training pipeline using Hugging Face TRL.
- Utilize QLoRA (4-bit) to ensure the model and training state fit within 8GB VRAM.
- Integrate the existing sandbox infrastructure into the RL reward loop.
- Provide a clear configuration interface via TOML.
- Implement a dry-run mode for rapid configuration testing.
- Track training and sandbox metrics using `trackio`.

**Non-Goals:**
- Multi-GPU or distributed training support.
- Full parameter fine-tuning (not feasible on 8GB VRAM).
- Training from scratch or on non-coding tasks.

## Decisions

### 1. Training Framework: TRL GRPOTrainer
We will use the `TRL` library's `GRPOTrainer`. 
- **Rationale**: It provides a high-level API for GRPO and natively supports PEFT/QLoRA, which is critical for our hardware.
- **Alternatives**: Implementing GRPO from scratch using PyTorch. *Rejected* due to complexity and the existence of a well-tested library.

### 2. VRAM Optimization Strategy
To fit into 8GB VRAM:
- **Quantization**: 4-bit NF4 quantization using `bitsandbytes`.
- **PEFT**: LoRA adapters with rank 32.
- **Precision**: `bfloat16` for compute.
- **Batch Size**: Small batch size with high gradient accumulation steps.
- **Liger Kernels**: Enabled to optimize memory and speed.
- **Rationale**: This combination is the industry standard for fine-tuning 1B+ parameter models on consumer hardware.

### 3. Reward Function Implementation
The reward function will be a custom Python callable passed to `GRPOTrainer`.
- **Extraction**: Uses regex to find triple-backtick code blocks.
- **Execution**: Calls the existing `Sandbox` infrastructure. It will attempt to batch executions where possible to reduce sandbox startup overhead.
- **Scoring**: Configurable between binary (all-or-nothing) and percentage-based rewards.
- **Rationale**: Leverages existing, tested sandbox code.

### 4. Configuration via TOML
- **Rationale**: TOML is more human-readable than JSON and more structured than flat environment variables. It allows for nested configurations (e.g., `[lora]`, `[sandbox]`).

### 5. Tracking with trackio
- **Rationale**: `trackio` is the recommended tracking tool for TRL and integrates well with Hugging Face Spaces for persistent experiment visualization.

### 6. Periodic Evaluation
During training, we want to periodically benchmark the model on HumanEvalPlus and MBPPPlus.
- **Rationale**: Relying solely on training loss/rewards can be deceptive. A custom `TrainerCallback` injects `TransformersEvaluator` into the training loop, capturing pass rates and securely logging them natively alongside training metrics. It runs based on the configured `eval_steps` and is safely bypassed during `dry_run`s.

## Risks / Trade-offs

- **[Risk] Out-of-Memory (OOM)** → **[Mitigation]** Use 4-bit quantization, disable `gradient_checkpointing` if memory allows or enable if needed, and keep the prompt/completion lengths within a reasonable budget (e.g., 512/4096).
- **[Risk] Slow Reward Calculation** → **[Mitigation]** Use `MontySandbox` as the primary engine for its low latency. Implement timeouts and batch processing.
- **[Risk] Improper Reward Shaping** → **[Mitigation]** Use `loss_type="dr_grpo"` and `scale_rewards` settings as requested to stabilize the RL signal.
- **[Risk] Data Leakage/Sandbox Security** → **[Mitigation]** Rely on the existing sandbox's isolation (Docker/Monty) and disable network access by default.
