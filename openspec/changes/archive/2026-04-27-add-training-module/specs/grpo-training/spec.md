## ADDED Requirements

### Requirement: GRPO Training Loop
The system SHALL implement a training loop using the `GRPOTrainer` from the TRL library, configured for Group Relative Policy Optimization (GRPO).

#### Scenario: Successful trainer initialization
- **WHEN** the training module is executed with a valid configuration file
- **THEN** the `GRPOTrainer` is initialized with the specified model, dataset, and training arguments

### Requirement: Verifiable Reward Function
The system SHALL implement a reward function that extracts code from model completions and verifies it by executing tests in a secure sandbox.

#### Scenario: Reward for passing code
- **WHEN** a model completion contains code that passes 100% of the associated test cases
- **THEN** the reward function returns a value of 1.0 (or the configured maximum)

#### Scenario: Zero reward for failing code
- **WHEN** a model completion contains code that fails one or more test cases, or cannot be extracted
- **THEN** the reward function returns 0.0

### Requirement: Sandbox Integration and Metrics
The training reward function SHALL use the project's `Sandbox` interface to execute code and SHALL log metrics about the execution (type, duration, errors) to the experiment tracker.

#### Scenario: Logging sandbox usage
- **WHEN** the reward function executes code in the sandbox
- **THEN** it records whether Monty or Docker was used and the execution duration in the `trackio` logs

### Requirement: Model Checkpointing and Uploading
The system SHALL save model checkpoints to a local directory and, if an `HF_TOKEN` is provided, upload the final trained adapters to the Hugging Face Hub.

#### Scenario: Local checkpointing
- **WHEN** the training reaches a configured checkpoint interval
- **THEN** the LoRA adapter weights are saved to the local output directory

### Requirement: Dry Run Mode
The system SHALL support a `--dry-run` flag that executes a minimal number of training steps (e.g., 10) to verify the end-to-end configuration without performing a full training run.

#### Scenario: Executing a dry run
- **WHEN** the training module is run with the `--dry-run` flag
- **THEN** it completes 10 training steps and saves the results to a temporary directory
