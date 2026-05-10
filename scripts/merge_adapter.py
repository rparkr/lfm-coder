"""Merge LoRA adapter with base model.

Run from the repository root:

```shell
uv run scripts/merge_adapter.py --upload
```
"""

import argparse
import os
import tomllib
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm_coder.logging_utils import get_logger

logger = get_logger("merge_adapter")

# 1. Load the base model and tokenizer
training_config = tomllib.load(Path("training_config.toml").open("rb"))
base_model_name = training_config["model_id"]
adapter_path = Path(training_config["output_dir"])
merged_model_path = adapter_path / "merged_model"
upload_repo = str(adapter_path) + "-merged"

logger.info(f"Loading base model: {base_model_name}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, dtype=torch.bfloat16, device_map="auto"
)
logger.info(f"Loading tokenizer: {base_model_name}")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# 2. Load and merge LoRA adapter
logger.info(f"Loading LoRA adapter from {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)

logger.info("Merging LoRA adapter with base model...")
model = model.merge_and_unload()

# 3. Save the final merged model
logger.info(f"Saving merged model to {merged_model_path}")
model.save_pretrained(merged_model_path)

logger.info(f"Saving tokenizer to {merged_model_path}")
# tokenizer can be None, so the type checker gets confused.
tokenizer.save_pretrained(merged_model_path)  # ty:ignore[unresolved-attribute]

print(f"Merged model saved to {merged_model_path}")

# 4. Upload to Hugging Face (optional)
# Set up argparse to handle --upload flag
parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model.")
parser.add_argument(
    "--upload", action="store_true", help="Upload merged model to Hugging Face"
)
args = parser.parse_args()

if not args.upload:
    print("Skipping upload. Use --upload to upload the merged model to Hugging Face.")
    exit(0)

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    logger.info(f"Uploading merged model to {upload_repo}")
    model.push_to_hub(upload_repo, token=hf_token)
    tokenizer.push_to_hub(upload_repo, token=hf_token)  # ty:ignore[unresolved-attribute]
    print(f"Merged model uploaded to {upload_repo}")
else:
    print("No HF_TOKEN environment variable set. Skipping upload.")
