import datetime
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer

from lfm_coder.device import detect_device, supports_quantization
from lfm_coder.evals.transformers_evaluator import TransformersEvaluator
from lfm_coder.logging_utils import get_logger
from lfm_coder.train.config import TrainingConfig

logger = get_logger(__name__)


def setup_trainer(
    config: TrainingConfig,
    train_dataset: Dataset,
    reward_funcs: list,
    dry_run: bool = False,
) -> GRPOTrainer:
    """
    Setup the GRPOTrainer with the given configuration and dataset.
    """
    # 1. Setup Tracking
    os.environ["TRACKIO_SPACE_ID"] = config.trackio_space_id
    os.environ["TRACKIO_PROJECT"] = config.trackio_project

    # 2. Resolve hardware backend
    device = detect_device(config.device)
    if config.use_quantization is None:
        use_quant = supports_quantization(device)
    else:
        use_quant = config.use_quantization
        if use_quant and not supports_quantization(device):
            logger.warning(
                f"use_quantization=True requested but device={device} does not support "
                "bitsandbytes; disabling quantization."
            )
            use_quant = False

    use_liger = config.use_liger_kernel and device == "cuda"
    optim = "paged_adamw_8bit" if use_quant else "adamw_torch"
    compute_dtype = (
        getattr(torch, config.bnb.bnb_4bit_compute_dtype)
        if device in ("cuda", "mps")
        else torch.float32
    )
    logger.info(
        f"Resolved hardware: device={device}, use_quant={use_quant}, "
        f"optim={optim}, liger={use_liger}, dtype={compute_dtype}"
    )

    # 3. Load Model and Tokenizer
    model_kwargs: dict = {"dtype": compute_dtype}
    if use_quant:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=config.bnb.load_in_4bit,
            bnb_4bit_quant_type=config.bnb.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.bnb.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb.bnb_4bit_compute_dtype),
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if not use_quant and device != "cpu":
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    # NOTE: Quiet the type checker; AutoTokenizer can return None, but won't in this case.
    if tokenizer.pad_token is None:  # ty:ignore[unresolved-attribute]
        tokenizer.pad_token = tokenizer.eos_token  # ty:ignore[unresolved-attribute, invalid-assignment]

    # 4. LoRA Config
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. GRPO Config
    max_steps = 10 if dry_run else -1

    run_name = config.run_name
    if not run_name:
        run_name = f"{config.model_id.split('/')[-1]}-grpo"

    if dry_run:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        run_name = f"dry-run-{run_name}-{timestamp}"

    grpo_config = GRPOConfig(
        optim=optim,
        output_dir=config.output_dir,
        learning_rate=config.learning_rate or 1e-5,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=True,  # re-calculate activations for backward pass to save memory
        max_completion_length=config.max_completion_length,
        num_generations=config.num_generations,
        temperature=config.temperature,
        eval_strategy="no",
        eval_steps=config.eval_steps,
        save_strategy="steps" if config.save_steps > 0 else "no",
        save_steps=config.save_steps,
        logging_steps=1,
        log_completions=True,
        max_steps=max_steps,
        report_to=["trackio"],
        run_name=run_name,
        loss_type=config.loss_type,
        use_liger_kernel=use_liger,
        dataloader_pin_memory=(device == "cuda"),
    )

    if hasattr(grpo_config, "scale_rewards"):
        grpo_config.scale_rewards = config.scale_rewards

    # 7. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    # 8. Setup Evaluation Callback
    if config.eval_steps > 0 and not dry_run:
        evaluator = TransformersEvaluator(
            model_name=config.model_id,
            model_id=f"{config.model_id.split('/')[-1]}-grpo",
            model=model,
            tokenizer=tokenizer,  # ty:ignore[invalid-argument-type]
            output_dir=Path(config.output_dir) / "eval_results",
            # batch_size=config.batch_size,
            max_tokens=config.max_completion_length,
            temperature=0.0,
        )

        class PeriodicEvalCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step > 0 and state.global_step % config.eval_steps == 0:
                    logger.info(
                        f"Running periodic evaluation at step {state.global_step}"
                    )
                    was_training = model.training
                    model.eval()
                    try:
                        results = evaluator.evaluate()
                        metrics = {}
                        for ds_name, ds_metrics in results.metrics.items():
                            metrics[f"eval/{ds_name}_pass_rate"] = ds_metrics.pass_rate

                        if metrics:
                            trainer.log(metrics)
                    except Exception as e:
                        logger.error(f"Periodic evaluation failed: {e}")
                    finally:
                        if was_training:
                            model.train()

        trainer.add_callback(PeriodicEvalCallback())

    return trainer
