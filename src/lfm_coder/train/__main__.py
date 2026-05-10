import os
from pathlib import Path

import typer

from lfm_coder.datasets.training_data import TrainingDataset
from lfm_coder.logging_utils import get_logger
from lfm_coder.train.config import load_config
from lfm_coder.train.rewards import CodingAccuracyReward
from lfm_coder.train.trainer import setup_trainer

app = typer.Typer(help="LFM-Coder Training CLI", no_args_is_help=True)
logger = get_logger(__name__)


@app.command(help="Train a model using GRPO", no_args_is_help=True)
def train(
    config_path: str = typer.Option(
        "training_config.toml", help="Path to the TOML configuration file"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Perform 10 training steps to verify setup"
    ),
    num_samples: int | None = typer.Option(
        None, help="Override number of samples to use"
    ),
):
    """
    Train a model using GRPO.
    """
    if not Path(config_path).exists():
        logger.error(f"Config file not found at {config_path}")
        raise typer.Exit(code=1)

    config = load_config(config_path)

    if num_samples is not None:
        config.num_train_records = num_samples

    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.warning(
            "HF_TOKEN environment variable is not set. "
            "Model will be saved locally but NOT uploaded to Hugging Face Hub. "
            "Set HF_TOKEN to enable automatic uploading."
        )

    # 1. Load Dataset
    logger.info(f"Loading dataset with {config.num_train_records} samples...")
    dataset_wrapper = TrainingDataset(
        seed=config.get_seed(), num_samples=config.num_train_records
    )
    train_dataset = dataset_wrapper.data

    # 2. Setup Reward Function
    reward_obj = CodingAccuracyReward(config)

    def coding_accuracy_reward(prompts, completions, tests, **kwargs):
        return reward_obj(
            prompts=prompts, completions=completions, tests=tests, **kwargs
        )

    # 3. Setup Trainer
    logger.info("Initializing trainer...")
    trainer = setup_trainer(
        config=config,
        train_dataset=train_dataset,
        reward_funcs=[coding_accuracy_reward],
        dry_run=dry_run,
    )

    # 4. Run Training
    logger.info(f"Starting training{' (DRY RUN)' if dry_run else ''}...")
    resume_checkpoint = False
    if config.resume_training and Path(config.output_dir).exists() and not dry_run:
        logger.info(f"Found existing training at {config.output_dir}. Resuming...")
        resume_checkpoint = True

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 5. Save and Push
    if dry_run:
        logger.info("Dry run complete. Saving to temporary directory...")
        trainer.save_model(str(Path(config.output_dir) / "dry_run_output"))
    else:
        logger.info("Training complete. Saving model...")
        trainer.save_model(config.output_dir)

        if hf_token:
            logger.info(f"Pushing model to {config.output_dir}...")
            trainer.push_to_hub()
        else:
            logger.info("Skipping push to hub (HF_TOKEN missing).")


if __name__ == "__main__":
    app()
