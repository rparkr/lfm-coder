## Repository Purpose

**lfm-coder** trains Liquid AI's LFM 2.5-1.2B-Instruct model using **GRPO with RLVR** (Reinforcement Learning from Verifiable Rewards). Trained via [TRL](https://github.com/huggingface/trl) to improve coding capabilities. Produces LoRA adapters for the base model.

- **Package manager**: `uv`
- **Type checker**: `ty`
- **Linter**: `ruff`
- **Test command**: `uv run pytest tests/`

**Use `scripts/` repository** for exploratory code (e.g., `scripts/benchmark_sandboxes.py` for sandbox troubleshooting).

## Developer Commands

- Install dev deps: `uv sync` (requires `container-pools` optional deps)
- Lint: `uv run ruff check src/`
- Type check: `uv run ty check src/`
- Run tests: `uv run pytest tests/`
- Run single test: `uv run pytest tests/<test_file>.py -v`

## Project Structure

- `src/lfm_coder/` — Main package (entry point)
  - `sandbox/` — Execution engines (`Sandbox`, `MontySandbox`, `DockerSandbox`)
  - `datasets/` — Training/eval dataset processing
  - `rewards/` — RLVR reward computation
  - `evals/` — Evaluation metrics/verification
- `scripts/` — Utility scripts (import mapping, sandbox demos, benchmarking)

## Key Architectural Facts

1. **Dual execution engines**:
   - `MontySandbox`: Fast Rust-based Python (pydantic-monty)
   - `DockerSandbox`: Full PyPI support via containers
   - Auto-fallback to Docker if Monty fails

2. **Training flow** (in progress):
   - Dataset: `OpenCoder-LLM/opc-sft-stage2` (SFT stage)
   - Evaluation: `humanevalplus`, `mbppplus`
   - Framework: TRL + QLoRA
   - Future: Config files for model/dataset selection

3. **Build quirks**:
   - `uv_build` includes `Dockerfile.sandbox` + `module_mapping.json` in build artifacts since those are used for DockerSandbox

4. **Dev workflow order**: `lint -> typecheck -> test`

## Repo-Specific Conventions

- Requires Python 3.13+
- Docker/Podman must be available to support the container-based sandboxes
- Test fixtures live in `tests/`

## Testing Quirks

- Sandbox tests verify auto-fallback behavior
- Reward helpers tested in `tests/test_helpers.py`
- The `PooledDockerSandbox` is not as reliable and fails often (which is why it is not recommended nor used in other areas of the codebase)
