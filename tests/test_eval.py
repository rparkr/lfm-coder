"""Tests for evaluation module."""

from unittest.mock import MagicMock, patch

from lfm_coder.evals.eval import Evaluator
from lfm_coder.evals.types import Checkpoint, GenerationResult, TaskResult


class MockEvaluator(Evaluator):
    """Mock evaluator for testing base class logic."""

    def generate(self, prompts, **kwargs):
        return [
            GenerationResult(
                completion="```python\ndef test():\n    return True\n```",
                token_count=10,
                generation_time=0.1,
                throughput=100.0,
            )
        ] * len(prompts)


def test_code_extraction():
    evaluator = MockEvaluator(model_name="test_model", model_id="test_model-baseline")

    # Test simple python block
    completion = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```"
    code, success = evaluator._extract_code(completion)
    assert code == "def add(a, b):\n    return a + b"
    assert success is True

    # Test last block
    completion = "First:\n```python\nold()\n```\nSecond:\n```python\nnew()\n```"
    code, success = evaluator._extract_code(completion)
    assert code == "new()"
    assert success is True

    # Test generic block
    completion = "```\nprint('hello')\n```"
    code, success = evaluator._extract_code(completion)
    assert code == "print('hello')"
    assert success is False

    # Test unclosed block
    completion = "Reasoning... ```python\ndef incomplete():"
    code, success = evaluator._extract_code(completion)
    assert code == "def incomplete():"
    assert success is False

    # Test no block
    completion = "No code here."
    code, success = evaluator._extract_code(completion)
    assert code is None
    assert success is False


def test_metric_calculation(tmp_path):
    evaluator = MockEvaluator(
        model_name="test_model", model_id="test_model-baseline", output_dir=tmp_path
    )
    results = [
        TaskResult(
            model_name=evaluator.model_name,
            model_id=evaluator.model_id,
            dataset_name="test",
            task_id="1",
            passed=True,
            correct_format=True,
            extraction_success=True,
            generation_time=2.0,
            sandbox_execution_time=1.0,
            token_count=100,
            throughput=50.0,
            prompt=[],
            completion="",
        ),
        TaskResult(
            model_name=evaluator.model_name,
            model_id=evaluator.model_id,
            dataset_name="test",
            task_id="2",
            passed=False,
            correct_format=True,
            extraction_success=True,
            generation_time=2.0,
            sandbox_execution_time=2.0,
            token_count=200,
            throughput=100.0,
            prompt=[],
            completion="",
        ),
        TaskResult(
            model_name=evaluator.model_name,
            model_id=evaluator.model_id,
            dataset_name="test",
            task_id="3",
            passed=False,
            correct_format=False,
            extraction_success=False,
            generation_time=0.0,
            sandbox_execution_time=0.0,
            token_count=0,
            throughput=0.0,
            prompt=[],
            completion="",
        ),
    ]

    checkpoint = Checkpoint(
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        checkpoint_file=tmp_path / "test_metrics.jsonl",
        dataset_name="test",
        completed_task_ids={"1", "2"},
        incomplete_task_ids={"3"},
        is_complete=False,
    )
    evaluator._save_results(results, checkpoint)

    metrics = evaluator._calculate_metrics("test", checkpoint)
    # Token count > 0 means it counts as a completed task in _calculate_metrics
    assert metrics.pass_rate == 1 / 2
    assert metrics.format_success_rate == 1.0
    assert metrics.extraction_success_rate == 1.0
    assert metrics.avg_sandbox_execution_time == 1.5
    assert metrics.completed_tasks == 2


def test_checkpoint_save_load(tmp_path):
    evaluator = MockEvaluator(
        model_name="test_model", model_id="test_model-baseline", output_dir=tmp_path
    )
    checkpoint_file = tmp_path / "test.jsonl"
    checkpoint = Checkpoint(
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        checkpoint_file=checkpoint_file,
        dataset_name="test",
        completed_task_ids=set(),
        incomplete_task_ids=set(),
        is_complete=False,
    )
    results = [
        TaskResult(
            model_name=evaluator.model_name,
            model_id=evaluator.model_id,
            task_id="1",
            dataset_name="test",
            prompt=[{"role": "user", "content": "hi"}],
            completion="hello",
            passed=True,
            token_count=10,
        ),
    ]
    evaluator._save_results(results, checkpoint)

    loaded = evaluator._load_results(checkpoint_file)
    assert len(loaded) == 1
    assert loaded[0].task_id == "1"
    assert loaded[0].prompt == [{"role": "user", "content": "hi"}]
    assert loaded[0].completion == "hello"
    assert loaded[0].passed is True


@patch("lfm_coder.evals.eval.HumanEvalPlusDataset")
@patch("lfm_coder.evals.eval.Sandbox")
def test_evaluate_dataset(mock_sandbox_cls, mock_ds_cls, tmp_path):
    mock_ds = MagicMock()
    mock_ds.data = MagicMock()
    mock_ds.data.filter.return_value = mock_ds.data
    mock_ds.data.iter.return_value = [
        {
            "task_id": ["T1"],
            "prompt": [[{"role": "user", "content": "p"}]],
            "test": ["{solution}"],
            "entry_point": ["f"],
        }
    ]
    mock_ds.data.__len__.return_value = 1
    mock_ds_cls.return_value = mock_ds

    mock_sandbox = MagicMock()
    mock_sandbox.run.return_value = [
        MagicMock(stdout='{"passed": true}', duration_sec=0.5, failed=False)
    ]
    mock_sandbox_cls.return_value = mock_sandbox

    evaluator = MockEvaluator(
        model_name="test_model", model_id="test_model-baseline", output_dir=tmp_path
    )
    checkpoint = Checkpoint(
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        checkpoint_file=tmp_path / "human_eval.jsonl",
        dataset_name="human_eval",
        completed_task_ids=set(),
        incomplete_task_ids=set(),
        is_complete=False,
    )
    metrics, results = evaluator._evaluate_dataset(
        dataset_name="human_eval",
        checkpoint=checkpoint,
    )

    assert len(results) == 1
    assert results[0].passed is True
    assert metrics.pass_rate == 1.0


@patch("lfm_coder.evals.eval.HumanEvalPlusDataset")
@patch("lfm_coder.evals.eval.Sandbox")
def test_evaluate_auto_new_run(mock_sandbox_cls, mock_ds_cls, tmp_path):
    # Mock dataset with 1 task
    mock_ds = MagicMock()
    mock_ds.data = MagicMock()
    # Mock filter to return itself if task_id not in completed_task_ids
    # In evaluate(), it calls _get_checkpoint first.
    mock_ds.data.filter.return_value = mock_ds.data
    mock_ds.data.iter.return_value = [
        {
            "task_id": ["T1"],
            "prompt": [[{"role": "user", "content": "p"}]],
            "test": ["{solution}"],
            "entry_point": ["f"],
        }
    ]
    mock_ds.data.__len__.return_value = 1
    mock_ds_cls.return_value = mock_ds

    mock_sandbox = MagicMock()
    mock_sandbox.run.return_value = [
        MagicMock(stdout='{"passed": true}', duration_sec=0.5, failed=False)
    ]
    mock_sandbox_cls.return_value = mock_sandbox

    evaluator = MockEvaluator(
        model_name="test_model", model_id="test_model-baseline", output_dir=tmp_path
    )

    # Create a "complete" checkpoint file with correct prefix
    checkpoint_file = tmp_path / f"{evaluator.model_id}_old.jsonl"
    checkpoint = Checkpoint(
        model_name=evaluator.model_name,
        model_id=evaluator.model_id,
        checkpoint_file=checkpoint_file,
        dataset_name="human_eval",
        completed_task_ids=set(),
        incomplete_task_ids=set(),
        is_complete=False,
    )
    evaluator._save_results(
        [
            TaskResult(
                model_name=evaluator.model_name,
                model_id=evaluator.model_id,
                task_id="T1",
                dataset_name="human_eval",
                prompt=[],
                completion="",
                passed=True,
                # token_count > 0 makes it "complete" because that means we have an LLM response
                # for the task.
                token_count=10,
            )
        ],
        checkpoint,
    )

    # Run evaluation with resume=True
    # It should detect the existing checkpoint is complete and start a NEW run
    result = evaluator.evaluate(dataset_names=["human_eval"], resume=True)

    # Check that a NEW file was created (in addition to the old one)
    checkpoint_files = list(tmp_path.glob(f"{evaluator.model_id}_*.jsonl"))
    assert len(checkpoint_files) == 2
    assert len(result.task_results) == 1
    assert result.task_results[0].task_id == "T1"
