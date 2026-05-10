<h1 align="center">LFM-Coder: High-Performance RLVR for Small Language Models</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3130/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a>
</p>

Fine-tune LLMs to enhance coding capabilities using **Reinforcement Learning from Verifiable Rewards** (RLVR) with **Group Relative Policy Optimization** (GRPO). Includes a **blazing-fast Python sandbox** for safely running model-generated code.

# Results

A model trained from this repository using only 1,000 examples from the [OpenCoder dataset](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) achieved a **49.1% improvement** in coding performance on the [MBPP benchmark](https://github.com/google-research/google-research/tree/master/mbpp) while maintaining general capabilities:

[![benchmark results showing the change in model performance after fine-tuning](./images/benchmark_results.svg)][benchmark-plot]

✨ _Try out the [trained model](https://huggingface.co/rparkr/LFM2.5-1.2B-Instruct-Coding), explore the [metrics during training](https://huggingface.co/spaces/rparkr/lfm-coder-training), or analyze the [training artifacts](https://huggingface.co/buckets/rparkr/lfm-coder-training-bucket/tree/README.md)._

# Why LFM-Coder?

Small language models (SLMs) are the key to fast, local coding agents, but they often struggle with complex programming tasks. Liquid AI's [LFM2.5-1.2B-Instruct](https://docs.liquid.ai/lfm/models/lfm25-1.2b-instruct) is exceptionally fast and efficient, but not optimized for coding out of the box.

LFM-Coder bridges this gap using **RLVR**. By training lightweight **LoRA adapters** (~22M parameters) with Hugging Face [**TRL**](https://github.com/huggingface/trl), we provide the model with a high-fidelity execution environment to learn from real-time, verifiable feedback. This approach significantly enhances coding performance while maintaining the model's tiny footprint and general capabilities.

# Key Innovations and Optimizations

This repository goes beyond basic fine-tuning by implementing a production-grade RLVR environment and training pipeline:

### 🚀 High-Performance Sandbox
- **Dual-Engine Architecture**: Seamlessly alternates between a blazing-fast Rust-based Python interpreter ([Monty](https://github.com/pydantic/monty/)) and full-featured Docker/Podman containers.
- **Massive Concurrency**: Threaded execution across all CPU cores for both engines, enabling high-throughput reward computation essential for GRPO.
- **Smart Dependency Management**: Packages are installed dynamically based on code requirements. Local caching ensures subsequent runs load instantaneously and can run without network access.
- **Enterprise-Grade Isolation**: Configurable resource guards (CPU/memory), execution timeouts, and network isolation to ensure secure execution of model-generated code.

### ⚡ Training and Evaluation Efficiency
- **Asynchronous Pipelining**: Overlaps GPU completion generation with CPU-based code verification to maximize hardware utilization and minimize idle time.
- **Optimized RLVR Pipeline**: Leverages QLoRA (4-bit) and Liger kernels to enable advanced GRPO training on consumer hardware (8GB VRAM).
- **Fault-Tolerant Workflows**: Robust state management with automatic resumption for both training and evaluation cycles.

### 📊 Data Quality and Integrity
- **Benchmark Sanitization**: Identifies and repairs incorrect test cases in standard benchmarks (HumanEvalPlus/MBPPPlus) to ensure rigorous evaluation.
- **Automated Validation**: Verifies all training examples against provided solutions to guarantee data quality before RLVR begins.
- **Granular Metrics**: Heuristic-driven extraction that calculates per-test-case pass rates and provides detailed logs for model weakness analysis.

# Getting Started: Training

### 1. Requirements
- **Hardware**: Single GPU with 8GB VRAM (e.g., RTX 4060).
- **Tooling**: [uv](https://github.com/astral-sh/uv#installation) installed.

### 2. Setup
```bash
git clone https://github.com/rparkr/lfm-coder.git && cd lfm-coder
export HF_TOKEN="your-hf-token"
```

### 3. Configuration
Update [`training_config.toml`](./training_config.toml) with your `model_id` and `output_dir`.

### 4. Run Training
```bash
# Dry run to verify configuration
uv run lfm-coder --dry-run

# Start full training
uv run lfm-coder
```

# Using the Python Sandbox

You can use the high-performance sandbox in your own projects for safe execution of LLM-generated code.

### Installation
```bash
uv add lfm-coder  # or pip install lfm-coder
```

### Basic Usage
The `Sandbox` class automatically routes code between Monty (fast) and Docker (full support).

```python
from lfm_coder.sandbox import Sandbox

sandbox = Sandbox()

# Batch execution (parallel)
results = sandbox.run(["1+1", "import math; math.sqrt(16)", "print('Hello')"])
for r in results:
    print(f"Stdout: {r.stdout} | Result: {r.result}")
```

### Advanced: Automatic Fallback

```python
code = """
import httpx  # Requires Docker fallback
r = httpx.get('https://example.com')
print(r.status_code)
"""
result = sandbox.run(code)
```

> [!NOTE]
> The Docker sandbox requires either [Podman](https://podman.io/) (recommended) or [Docker](https://docs.docker.com/engine/) to be installed and running.

# Project Roadmap and Stats

## 🗺️ Status
- [x] **Dual Sandboxes**: `MontySandbox` + `DockerSandbox` with auto-routing.
- [x] **Data Pipeline**: Automated sampling, verification, and repair of benchmarks.
- [x] **RLVR Training**: GRPO integration with TRL and GPU optimizations.
- [x] **Evaluation**: Scoring module with GPU/CPU pipelining.
- [ ] **Ollama support**: Fix chat template in [fine-tuned GGUF model](https://huggingface.co/rparkr/LFM2.5-1.2B-Instruct-Coding-merged-Q4_K_M-GGUF) for multi-turn chat.

## 📊 Training Performance Metrics

| Metric | Monty Sandbox (Rust) | Docker Sandbox (Container) |
| :--- | :--- | :--- |
| **Execution Count** | 18,556 (77.3%) | 5,444 (22.7%) |
| **Avg. Speed** | **1.01 ms** | 2,577 ms |
| **Median Speed** | **0.4 ms** | 2,240 ms |
| **Success Rate** | 69.8% | 35.8% |
| **Throughput** | ~1,000 exec/sec | ~0.4 exec/sec |

*Monty execution is **2,000x - 5,000x faster** than the Docker fallback, providing the massive throughput required for efficient RLVR training.*

# Acknowledgments

- [pydantic-monty](https://github.com/pydantic/monty/) for the lightning-fast Python sandbox.
- [TRL](https://github.com/huggingface/trl) and [trackio](https://github.com/gradio-app/trackio) for the RL framework and monitoring.
- [Evalplus](https://huggingface.co/datasets/evalplus) for the benchmark datasets.
- [OpenCoder-LLM](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2) for training data.
- [Liquid AI](https://docs.liquid.ai/lfm/getting-started/welcome) for the LFM2.5 model and GRPO guidance.

# License

Code: [MIT license](./LICENSE).  
Model Weights: [LFM license](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE) (Commercial restriction for >$10M revenue orgs).

<!-- Footnotes and links -->

[benchmark-plot]: https://vega.github.io/editor/#/url/vega-lite/N4IgJAzgxgFgpgWwIYgFwhgF0wBwqgegIDc4BzJAOjIEtMYBXAI0poHsDp5kTykBaADZ04JAGyUALJQCMlAFYQ2AOxAAaEEyRQA1mQBObBsoAmaTYO071IKCoBmNMmlBIAHjQguQBmmdT2SIIQcBqWTHCCAILKZIJwaAAMAL5h5HCm3uGRAMJsgmz65gDuMCI2mHTxeQVF6KXlqSCVmPHedrUlZZgJTcQ0cMXtKpXKDEYQABJwTlhoAMyJiRp2yqPjDBAA6n70C0sajoKC5kyWujYQmIY6Cahjx8lNJkiYKKigykgIdyAvb-x5pJ5mITKC4ABOSRwJAADjEUPscHsUFhSEScCg9nsEKQEKgAFYQM9XkgQpgvB8-qTAcDQeCoTD4Yjkaj0Zjsbj8UTUABtUARZSwZD6azoSYMZDKAAEAFFiEEbLAkLE7g9BBoEGwTJFTmSEhocGSIAABGRJSgEgBMMlSAoywqQovMEqlcoVJxWMBVZDu-ESlESVvmmu1uvQjmUcH4mGMcDMhuNZotBMSELtmgdMBFYpAAFkAEIABSLSu9qrQ6tDOpO6C0IRsRogpvNqAD8xkYgzgsdzvQhZLZZ9dwDkghMmr4ZAkejsajCZATZbFskAHZbQBdJrwWaYcyrN40KNFMJIACecDq-JADu1R+cVI6hW8jki-hAWprlygQTuoBMbDIEeaC8po+o2DOMZxmYG4aPow4gSAADEJhWqhJiSDYSEyPYq6rkwmFbhomBnjgvzKIBR6Kk0bBGlAdBnsMph0OwqhUka8EIOYHHfAA+rCTKwkwuJQFaUCrkgcAyEwNgegwdy2hockjpQ8xNJgbD5JUOAgaAr6CO+n66hm+nvj22ZOtYxFVL8BZZjmxJqHpAwGdxSbmochTIHu6BqKp9gVDZbnNma0raFADDwVAjEmS5hlhicsVvvu5a+hBXmvOYADUsgAKTErBIBuC+cWnPZlmXIUPl-HA0AZCY96Ba0tnlc6xGkeRlFfIlGhuAA8ti5Ilcl6BGZ6zQdeYFEIFRPUgIxVKmcFy7Wc1y2heFkXaIx7VkeYACODAqi0rw0KQxJNA5VIkXtdZOo5IBfD85j9IMvFwGIoIyMiTBiJI2g4b9vGJI5oC3g1sTDJ0VLKV0jQaLR2gMUxDWVCo3g8Vx6CY-xgnCdoYkSVJMlKUE8loIpICw22qnqXAbg+c5I2Lu56X6N55h5Q9GlaTQOl8kzrmjQloPTqVdatVZzRBegdlChZzpJULLMhR504ZdVfnzAFq1tNj7lhVAEVRTFTli8zY2i0t6DKhWnns5l6A5TI+XJIVxWLeLmbyw5GhKPo1U6nVzGQ7rLU+xVu2dTN3UPf1g1wIz5vK5bUdTV11EaAtgvvkuyZh+tMiG8b20VJN6CHcddCnedTyahVVL1pER6-EwbDYIBNgmAt-AEp5axbDMZBzHW+QLjdvw9AzF2FZjlLXkwR7vvEvqZBoT2-DjAlwkJImE5J0mXJEmJJ6Z88fiLhXo8LmxwGwpAnhNt2LmwR57n0AzFOfr3FO9n0mN9ewv1-pQEBmIYGIAtyFRaHrEAAAZAAYnmK0lp+BHiuPoBgUBMDSgABRNwAJTSmIBASg0pEHINQeg64WDMD8DsDqfQeDIJznjAQmwt92aJ3eGDBAEQTB9RwGjZQlJgB1xAMUXYMB9wjCQC3IoyQgA/view
