import random
import tomllib
from typing import Any

from pydantic import BaseModel, Field

from lfm_coder.sandbox import SandboxType


class SandboxConfig(BaseModel):
    """Configuration for the sandbox environment used during reward calculation."""

    type: SandboxType = SandboxType.AUTO
    max_memory_mb: int = 64
    max_execution_time_sec: int = 10
    network_access: bool = False
    use_cache: bool = True


class LoraConfigModel(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) adapters."""

    rank: int = 32
    alpha: int = 32
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    ]


class BitsAndBytesConfigModel(BaseModel):
    """Configuration for 4-bit quantization using BitsAndBytes (QLoRA)."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class RewardConfig(BaseModel):
    """Configuration for the verifiable reward function."""

    # If True, 1.0 if all tests pass, else 0.0. If False, percentage of tests passed.
    binary_reward: bool = False


class TrainingConfig(BaseModel):
    """Root configuration for the GRPO training process."""

    # Model settings
    model_id: str
    # New model repository name for Hugging Face
    output_dir: str

    # Hardware
    device: str = "auto"  # "auto" | "cuda" | "mps" | "cpu"
    # None means auto: on for CUDA, off for MPS/CPU.
    use_quantization: bool | None = None

    # Dataset and sampling
    seed: int | None = None
    num_train_records: int = 10_000

    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # Number of completions generated per prompt (G in GRPO)
    num_generations: int = 8
    # Learning rate for training (defaults to TRL's default if not set)
    learning_rate: float | None = None
    temperature: float = 0.5
    max_completion_length: int = 4096

    # Run config
    run_name: str | None = None
    resume_training: bool = True

    # Intervals
    eval_steps: int = 100
    save_steps: int = 10

    # GRPO Specifics
    loss_type: str = "dr_grpo"
    # Can be "batch" or False
    scale_rewards: Any = "batch"
    use_liger_kernel: bool = True

    # LoRA and Quantization
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    bnb: BitsAndBytesConfigModel = Field(default_factory=BitsAndBytesConfigModel)

    # Sandbox
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)

    # Rewards
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    # Tracking
    trackio_space_id: str = "rparkr/lfm-coder-training"
    trackio_project: str = "lfm-coder-training"

    def get_seed(self) -> int:
        """Get the random seed, generating a new one if it is None."""
        if self.seed is None:
            return random.randint(0, 2**32 - 1)
        return self.seed


def load_config(path: str) -> TrainingConfig:
    """Load the training configuration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return TrainingConfig(**data)
