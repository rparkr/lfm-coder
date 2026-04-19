# lfm-coder
GRPO with RLVR training Liquid AI's LFM 2.5-instruct model to enhance coding capabilities.

---

Fine-tune an LLM to enhance its coding abilities through **RLVR** (Reinforcement Learning from Verifiable Rewards) using [**TRL**](https://github.com/huggingface/trl).

Currently, this project trains [**Liquid AI's LFM 2.5-Instruct**](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) model — a lightweight, fast, hybrid model that is not strong at coding out of the box.

The training produces **LoRA adapters** that can be combined with the base model to improve its coding abilities while retaining the base model performance in other areas.

The goal of this project is to create a model that is decently good at coding while remaining incredibly fast and able to run on consumer hardware as a simple coding assistant.

# RL Environment

In RLVR, the reinforcement learning environment is the central part of training, enabling the model to improve with online data by receiving real-time rewards from the environment.

## Features
- **Dual Execution Engines**: Uses [pydantic-monty](https://github.com/pydantic/monty/) (lightning-fast, Rust-based Python interpreter) for standard Python code and Docker/Podman containers for full Python support.
- **Automatic Fallback**: Transparently switches to Docker if Monty doesn't support a specific module or feature.
- **Concurrent Batch Execution**: Execute multiple code snippets in parallel for high-throughput reward computation during training -- essential for Group Relative Policy Optimization (GRPO).
- **Third-Party Libraries**: Docker sandbox supports importing any PyPI library (numpy, polars, PyTorch, etc.)
- **Resource Guards and Network isolation**: Configurable memory limits, execution timeouts, and CPU constraints to prevent runaway code


## Example sandbox usage

```python
from lfm_coder.sandbox import Sandbox

sb = Sandbox()

# Single execution
result = sb.run("print('Hello World')")
print(result.stdout)

# Batch execution (parallel)
results = sb.run(["1+1", "2+2", "3+3"])
for r in results:
    print(r.output)
```

---

# Datasets

This project uses the following HuggingFace datasets for training and evaluation:

- **Training**: [OpenCoder-LLM/opc-sft-stage2](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2): Educational instruction dataset for coding
- **Evaluation**:
    - [evalplus/humanevalplus](https://huggingface.co/datasets/evalplus/humanevalplus): Enhanced HumanEval benchmark from OpenAI more test cases
    - [evalplus/mbppplus](https://huggingface.co/datasets/evalplus/mbppplus/): enhanced version of the Mostly Basic Python Programs benchmark from Google with more test cases

---

# Project Status

## ✅ Implemented

- [x] **Coding Sandboxes**: `MontySandbox`, `DockerSandbox`, and auto-routing `Sandbox` wrapper
- [x] **Training Dataset Processing**: Sampling, augmenting instructions, formatting in messages format, verifying correctness
- [x] **Evaluation Dataset Processing**: Processing and verification of evaluation data
- [x] **Reward Helper Functions**: Helper functions for RLVR reward computation
- [x] **Test Suite**: Coverage of sandbox methods and reward helpers

## 🚧 Coming soon

- [ ] **Evaluation Module**: Write evaluation module for scoring the model on the evaluation dataset (benchmarks)
- [ ] **Baseline Performance**: Establish the model's baseline performance on the evaluation sets
- [ ] **Training Script**: Create training script using TRL
- [ ] **Checkpointing & Metrics**: Add checkpointing and metric collection (evaluation set performance)
- [ ] **Run Training**: Execute the training process
- [ ] **Publish Results**: Document and publish training results
- [ ] **Config Files**: Set up configuration files to control the training process (e.g., which model to train, LoRA parameters, training dataset size)

---

# License

MIT
