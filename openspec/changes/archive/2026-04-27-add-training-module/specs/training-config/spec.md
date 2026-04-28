## ADDED Requirements

### Requirement: TOML Configuration File
The system SHALL read training parameters from a TOML configuration file located in the project root.

#### Scenario: Valid configuration loading
- **WHEN** the training script starts and finds `training_config.toml`
- **THEN** it parses the file and applies the settings to the model, trainer, and sandbox

### Requirement: Default Parameter Management
The system SHALL provide sensible default values for all configuration parameters if they are missing from the TOML file.

#### Scenario: Using default seed
- **WHEN** the `seed` parameter is missing from the configuration
- **THEN** the system generates a random seed and logs it

### Requirement: QLoRA and PEFT Configuration
The configuration SHALL support detailed settings for QLoRA (4-bit quantization) and PEFT (LoRA rank, alpha, target modules) to optimize VRAM usage.

#### Scenario: Applying LoRA settings
- **WHEN** the configuration specifies `lora_rank = 32` and `lora_alpha = 32`
- **THEN** the model is initialized with a `LoraConfig` using those values

### Requirement: Hardware Optimization Defaults
The system SHALL default to settings that are stable on 8GB VRAM hardware, including 4-bit quantization and gradient accumulation.

#### Scenario: VRAM-efficient defaults
- **WHEN** no hardware-specific overrides are provided
- **THEN** the system defaults to `load_in_4bit=True` and `bnb_4bit_compute_dtype="bfloat16"`
