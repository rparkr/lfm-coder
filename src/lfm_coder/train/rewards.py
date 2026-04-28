import json
from typing import Any

from lfm_coder.logging_utils import get_logger
from lfm_coder.rewards.utils import extract_code, pass_rate
from lfm_coder.sandbox import Sandbox, SandboxExecution
from lfm_coder.train.config import TrainingConfig

logger = get_logger(__name__)


class CodingAccuracyReward:
    __name__ = "coding_accuracy_reward"

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.sandbox = Sandbox(
            max_duration_sec=config.sandbox.max_execution_time_sec,
            max_memory_mb=config.sandbox.max_memory_mb,
            disable_network=not config.sandbox.network_access,
            use_cache=config.sandbox.use_cache,
            sandbox_type=config.sandbox.type,
        )

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        tests: list[str],
        **kwargs: Any,
    ) -> list[float]:
        """
        Reward function for GRPOTrainer.

        Args:
            prompts: List of model prompts.
            completions: List of model completions (strings or message dicts).
            tests: List of test scripts from the dataset.
            **kwargs: Other dataset columns and TRL training args.

        Returns:
            List of rewards (floats).
        """
        rewards = []
        code_to_run = []
        indices = []

        # 1. Extract code and prepare batch
        for i, (completion, test_script) in enumerate(zip(completions, tests)):
            if isinstance(completion, list):
                completion_text = completion[-1]["content"] if completion else ""
            else:
                completion_text = completion

            code, is_correct_format = extract_code(completion_text)

            if code is None:
                rewards.append(0.0)
                continue

            # Combine code with test script
            # Note: the test script expects the function to be defined in the same scope.
            full_code = f"{code}\n\n{test_script}"
            code_to_run.append(full_code)
            indices.append(i)
            rewards.append(0.0)  # Placeholder

        if not code_to_run:
            return rewards

        # 2. Execute batch in sandbox
        try:
            executions: list[SandboxExecution] = self.sandbox.run(
                code=code_to_run,
                skip_compatibility_check=True,  # Unified Sandbox handles fallback
            )
        except Exception as e:
            logger.error(f"Sandbox execution failed: {e}")
            return rewards

        # 3. Parse results and calculate rewards
        for idx, execution in zip(indices, executions):
            # Log metrics if tracking is enabled
            # We can log to a global tracker or just rely on TRL's logging
            # For now, we'll just log to our logger
            logger.debug(
                f"Execution result: sandbox={execution.sandbox_type}, "
                f"duration={execution.duration_sec:.4f}s, "
                f"success={execution.success}"
            )

            if execution.failed:
                rewards[idx] = 0.0
                continue

            try:
                # The test script prints a JSON list of booleans as the last line
                output_parts = execution.stdout.strip().split("\n")
                if not output_parts:
                    rewards[idx] = 0.0
                    continue

                results = json.loads(output_parts[-1])
                if not isinstance(results, list):
                    rewards[idx] = 0.0
                    continue

                p_rate = pass_rate(results)

                if self.config.rewards.binary_reward:
                    rewards[idx] = 1.0 if p_rate == 1.0 else 0.0
                else:
                    rewards[idx] = p_rate

            except (json.JSONDecodeError, IndexError) as e:
                logger.error(
                    f"Failed to parse sandbox output: {e}. Output: {execution.stdout}"
                )
                rewards[idx] = 0.0

        return rewards
