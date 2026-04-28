## 1. Setup and Dependencies

- [x] 1.1 Ensure that `trl[peft]`, `liger-kernel`, `bitsandbytes`, and `trackio` are in `pyproject.toml`
- [x] 1.2 If needed, run `uv sync` to install new dependencies
- [x] 1.3 Create the `src/lfm_coder/train/` package directory

## 2. Configuration System

- [x] 2.1 Implement `src/lfm_coder/train/config.py` to parse TOML using `pydantic` or `dataclasses`
- [x] 2.2 Create a default `training_config.toml` in the project root with the specified defaults (model paths, LoRA settings, sandbox params)
- [x] 2.3 Implement logic to warn about missing `HF_TOKEN` for model uploads

## 3. Verifiable Reward Function

- [x] 3.1 Reuse `extract_code` from `src/lfm_coder/rewards/utils.py`
- [x] 3.2 Implement the reward function callable that uses `Sandbox.run(code_batch)`
- [x] 3.3 Add support for configurable scoring (binary vs. percentage correct)
- [x] 3.4 Integrate sandbox metric logging into the reward function

## 4. Training Core Implementation

- [x] 4.1 Implement `src/lfm_coder/train/trainer.py` for model loading and trainer setup
- [x] 4.2 Configure `BitsAndBytesConfig` for 4-bit quantization as specified
- [x] 4.3 Configure `LoraConfig` with specified rank, alpha, and target modules
- [x] 4.4 Initialize `GRPOTrainer` with the custom reward function and `GRPOConfig` (loss_type, scaling, etc.)
- [x] 4.5 Integrate `trackio` for experiment logging to Hugging Face Spaces
- [x] 4.6 Ensure that all logged metrics from TRL are included

## 5. Execution and CLI

- [x] 5.1 Implement the main entry point in `src/lfm_coder/train/__main__.py` using the `typer` library.
- [x] 5.2 Implement the `--dry-run` flag to execute 10 steps and save to a temporary directory
- [x] 5.3 Add a CLI command or update `pyproject.toml` scripts to run the trainer

## 6. Testing and Documentation

- [x] 6.1 Add unit tests for the reward function extraction and scoring logic
- [x] 6.2 Verify the dry-run execution on the local 8GB VRAM hardware
- [x] 6.3 Update the project README with training instructions and notes on `HF_TOKEN`

## 7. Periodic Evaluation & Dynamic Config (Additional)

- [x] 7.1 Implement dynamic `run_name` logic to prevent dry-runs from overwriting true runs
- [x] 7.2 Add configurable `resume_training` to explicitly load the latest checkpoint
- [x] 7.3 Use a custom `TrainerCallback` to integrate `TransformersEvaluator` into the training loop
- [x] 7.4 Disable evaluation runs during `--dry-run` to improve diagnostic speed
