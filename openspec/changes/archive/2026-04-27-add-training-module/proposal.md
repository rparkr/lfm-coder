## Why

The `lfm-coder` project requires a dedicated training module to fine-tune the LFM 2.5-1.2B-Instruct model using Group Relative Policy Optimization (GRPO) with Reinforcement Learning from Verifiable Rewards (RLVR). This is essential to improve the model's coding capabilities by rewarding it for generating code that passes functional tests in a secure sandbox environment.

## What Changes

- **New Training Module**: Implementation of a training script/module using the Hugging Face TRL library.
- **Custom Reward Function**: A reward mechanism that extracts code from model completions, executes it in the project's existing sandbox, and calculates rewards based on test pass rates.
- **Configuration System**: A TOML-based configuration file in the project root to manage model paths, hyperparameters, LoRA settings, and sandbox parameters.
- **Experiment Tracking**: Integration with `trackio` for logging training progress, evaluation metrics, and sandbox execution data to Hugging Face Spaces.
- **Dry Run Capability**: A flag to perform a minimal number of training steps to validate the environment and configuration.
- **Sandbox Metrics Logging**: Enhancement of the sandbox interaction during training to log execution details (type, duration, errors) for debugging.

## Capabilities

### New Capabilities
- `grpo-training`: Implementation of the GRPO training loop, including integration with `GRPOTrainer`, model loading with QLoRA, and reward calculation logic.
- `training-config`: A centralized configuration system using TOML to manage all training, hardware, and sandbox parameters.

### Modified Capabilities
- `transformers-evaluation`: While the core evaluation logic remains similar, the training module will trigger evaluation runs using new `model_id` values per step.

## Impact

- **Dependencies**: Adds `trl[peft]`, `liger-kernel`, `bitsandbytes`, `trackio`, and `typer` to the project.
- **Source Code**: New package `src/lfm_coder/train/` and updates to existing modules to support training-specific logging.
- **Hardware**: Optimized for 8GB VRAM environments using 4-bit quantization and specific gradient accumulation settings.
- **Environment**: Requires `HF_TOKEN` for model uploads and `TRACKIO_*` variables for experiment tracking.
