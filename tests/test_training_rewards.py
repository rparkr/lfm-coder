from unittest.mock import patch

import pytest

from lfm_coder.sandbox.types import SandboxError, SandboxExecution, SandboxType
from lfm_coder.train.config import RewardConfig, SandboxConfig, TrainingConfig
from lfm_coder.train.rewards import CodingAccuracyReward


@pytest.fixture
def mock_config():
    return TrainingConfig(
        model_id="test-model",
        output_dir="test-output",
        rewards=RewardConfig(binary_reward=False),
        sandbox=SandboxConfig(type=SandboxType.MONTY),
    )


@pytest.fixture
def reward_fn(mock_config):
    return CodingAccuracyReward(mock_config)


def test_reward_extraction_failure(reward_fn):
    # Completion with no code block
    completions = ["This is just text with no code."]
    tests = ["assert True"]

    rewards = reward_fn(prompts=[""], completions=completions, tests=tests)
    assert rewards == [0.0]


def test_reward_parsing_success(reward_fn):
    # Completion with valid code block
    completions = ["```python\ndef add(a, b): return a + b\n```"]
    tests = ["import json; print(json.dumps([True, True, False]))"]

    mock_execution = SandboxExecution(
        stdout="[true, true, false]",
        stderr="",
        sandbox_type=SandboxType.MONTY,
        duration_sec=0.1,
        is_valid_python=True,
    )

    with patch("lfm_coder.sandbox.Sandbox.run", return_value=[mock_execution]):
        rewards = reward_fn(prompts=[""], completions=completions, tests=tests)
        # 2 out of 3 tests pass = 0.666...
        assert rewards[0] == pytest.approx(0.6666, abs=1e-4)


def test_reward_binary_mode(mock_config):
    mock_config.rewards.binary_reward = True
    reward_fn = CodingAccuracyReward(mock_config)

    completions = ["```python\ndef add(a, b): return a + b\n```"]
    tests = ["import json; print(json.dumps([True, True, False]))"]

    mock_execution = SandboxExecution(
        stdout="[true, true, false]",
        stderr="",
        sandbox_type=SandboxType.MONTY,
        duration_sec=0.1,
        is_valid_python=True,
    )

    with patch("lfm_coder.sandbox.Sandbox.run", return_value=[mock_execution]):
        rewards = reward_fn(prompts=[""], completions=completions, tests=tests)
        # In binary mode, if any test fails, reward is 0.0
        assert rewards[0] == 0.0


def test_reward_execution_failure(reward_fn):
    completions = ["```python\ndef add(a, b): return a + b\n```"]
    tests = ["import json; print(json.dumps([True]))"]

    mock_execution = SandboxExecution(
        stdout="",
        stderr="SyntaxError",
        errors=[SandboxError("Syntax error")],
        sandbox_type=SandboxType.MONTY,
        duration_sec=0.1,
        is_valid_python=True,
    )

    with patch("lfm_coder.sandbox.Sandbox.run", return_value=[mock_execution]):
        rewards = reward_fn(prompts=[""], completions=completions, tests=tests)
        assert rewards[0] == 0.0
