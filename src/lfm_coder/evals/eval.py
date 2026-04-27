"""Base evaluator class.

This module provides the base Evaluator class and the TransformersEvaluator implementation.
It supports evaluation on HumanEvalPlus and MBPPPlus datasets.

Example usage:
    # Transformers evaluation
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lfm_coder.evals.eval import TransformersEvaluator

    model = AutoModelForCausalLM.from_pretrained("path/to/model")
    tokenizer = AutoTokenizer.from_pretrained("path/to/model")
    evaluator = TransformersEvaluator(model, tokenizer)
    results = evaluator.evaluate(dataset_names=["human_eval"])
    print(f"Pass rate: {results.metrics['human_eval'].pass_rate}")
"""

from __future__ import annotations

import concurrent.futures
import datetime
import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from lfm_coder import rewards
from lfm_coder.datasets.eval_data import HumanEvalPlusDataset, MBPPPlusDataset
from lfm_coder.evals.types import (
    Checkpoint,
    DatasetMetrics,
    EvaluationResult,
    GenerationResult,
    TaskResult,
)
from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox import Sandbox, SandboxType

logger = get_logger(__name__)


class Evaluator(ABC):
    """Base class for model evaluation.

    Attributes:
        model_name: Name of the model being evaluated. This is used to load the model.
        model_id: Name to be used to identify the model in the result logs.
            Checkpoints are based on this model ID.
        output_dir: Directory to save evaluation results.
        batch_size: Number of samples to process in a batch.
        temperature: Temperature for model generation.
        max_tokens: Maximum tokens to generate.
        sandbox_type: Type of sandbox to use for code execution.
    """

    def __init__(
        self,
        model_name: str,
        model_id: str,
        output_dir: str | Path = "data/evaluation/results",
        batch_size: int = 16,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        sandbox_type: SandboxType = SandboxType.DOCKER,
    ):
        self.model_name = model_name
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_prefix = f"{self.model_id}_"
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.sandbox = Sandbox(
            sandbox_type=sandbox_type,
            use_cache=True,
            disable_network=False,
            max_duration_sec=60.0,
        )

    @abstractmethod
    def generate(
        self, prompts: list[list[dict[str, str]]], **kwargs: Any
    ) -> list[GenerationResult]:
        """Generate completions for a list of prompts.

        Args:
            prompts: List of message-formatted prompts.
            **kwargs: Additional generation parameters.

        Returns:
            List of GenerationResult objects.
        """
        pass

    def evaluate(
        self,
        dataset_names: list[str] | None = None,
        resume: bool = True,
    ) -> EvaluationResult:
        """Run evaluation on specified datasets.

        Args:
            dataset_names: List of datasets to evaluate ("human_eval", "mbpp").
            resume: Whether to resume from an existing checkpoint.

        Returns:
            Overall evaluation results.
        """
        if dataset_names is None:
            dataset_names = ["human_eval", "mbpp"]

        datasets_to_evaluate = set(dataset_names)

        checkpoints: dict[str, Checkpoint] = {}

        for dataset_name in dataset_names:
            checkpoints[dataset_name] = self._get_checkpoint(resume, dataset_name)
            if checkpoints[dataset_name].is_complete:
                datasets_to_evaluate.remove(dataset_name)

        if len(datasets_to_evaluate) == 0:
            logger.info(
                f"[{self.model_id}] All datasets are complete; starting a new checkpoint."
            )
            # Add back all datasets to be evaluated and get a fresh checkpoint file.
            datasets_to_evaluate.update(dataset_names)
            for dataset_name in dataset_names:
                checkpoints[dataset_name] = self._get_checkpoint(
                    resume=False,
                    dataset_name=dataset_name,
                )

        result = EvaluationResult(model_name=self.model_name)

        for dataset_name in datasets_to_evaluate:
            logger.info(
                f"[{self.model_id}] Starting evaluation on dataset: {dataset_name}"
            )
            dataset_metrics, task_results = self._evaluate_dataset(
                dataset_name=dataset_name,
                checkpoint=checkpoints[dataset_name],
            )
            result.metrics[dataset_name] = dataset_metrics
            result.task_results.extend(task_results)

            logger.info(
                f"[{self.model_id}] Finished evaluation on {dataset_name}. "
                f"Pass rate: {dataset_metrics.pass_rate:.1%}"
            )

        return result

    def _get_checkpoint(self, resume: bool, dataset_name: str) -> Checkpoint:
        """
        Resolve the run ID and whether to resume from the lastest checkpoint file.

        Args:
            resume: Whether to resume from an existing checkpoint.
            dataset_name: Name of the dataset to evaluate.

        Returns:
            Checkpoint object.
        """

        # Get all checkpoint files for this model, sorted by creation time.
        checkpoint_files = sorted(
            self.output_dir.glob(f"{self.checkpoint_prefix}*.jsonl")
        )

        # If resume is False OR no checkpoint files are found, create a new checkpoint.
        if not resume or not checkpoint_files:
            timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H-%M-%SZ"
            )
            checkpoint_file = (
                self.output_dir / f"{self.checkpoint_prefix}{timestamp_str}.jsonl"
            )
            return Checkpoint(
                model_name=self.model_name,
                model_id=self.model_id,
                checkpoint_file=checkpoint_file,
                dataset_name=dataset_name,
                completed_task_ids=set(),
                incomplete_task_ids=set(),
                is_complete=False,
            )
        # If resume is True AND there are checkpoint files, determine which tasks
        # remain for that dataset.

        # Load the latest checkpoint file.
        latest_checkpoint = checkpoint_files[-1]
        existing_results = list(
            filter(
                lambda result: result.dataset_name == dataset_name,
                self._load_results(latest_checkpoint),
            )
        )

        # Determine which tasks remain for that dataset by checking which ones have
        # a response from the LLM.
        completed_task_ids = set()
        incomplete_task_ids = set()
        for result in existing_results:
            if result.token_count and result.token_count > 0:
                completed_task_ids.add(result.task_id)
            else:
                incomplete_task_ids.add(result.task_id)

        is_complete = (
            len(existing_results) > 0
            and len(incomplete_task_ids) == 0
            and len(completed_task_ids) == len(existing_results)
        )

        if is_complete:
            logger.info(
                f"[{self.model_id}] Latest checkpoint is complete, nothing to resume. "
                f"Dataset: {dataset_name}. Checkpoint: {latest_checkpoint.name}"
            )
        else:
            logger.info(
                f"[{self.model_id}] Resuming from latest checkpoint. "
                f"Dataset: {dataset_name}. Checkpoint: {latest_checkpoint.name}"
            )

        return Checkpoint(
            model_name=self.model_name,
            model_id=self.model_id,
            checkpoint_file=latest_checkpoint,
            dataset_name=dataset_name,
            completed_task_ids=completed_task_ids,
            incomplete_task_ids=incomplete_task_ids,
            is_complete=is_complete,
        )

    def _evaluate_dataset(
        self,
        dataset_name: str,
        checkpoint: Checkpoint,
    ) -> tuple[DatasetMetrics, list[TaskResult]]:
        """Evaluate a single dataset."""
        if dataset_name == "human_eval":
            ds_wrapper = HumanEvalPlusDataset()
        elif dataset_name == "mbpp":
            ds_wrapper = MBPPPlusDataset()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset = ds_wrapper.data
        pending_tasks = dataset.filter(
            lambda x: x["task_id"] not in checkpoint.completed_task_ids
        )

        task_results: list[TaskResult] = []

        # Use a single-worker executor to overlap LLM inference (GPU) with sandbox execution (CPU)
        # and thus increase overall throughput.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            prev_future = None

            for batch in tqdm(
                pending_tasks.iter(batch_size=self.batch_size),
                desc=f"[{self.model_id}] Evaluating {dataset_name}",
                total=math.ceil(len(pending_tasks) / self.batch_size),
            ):
                prompts = batch["prompt"]

                # 1. Generation (synchronous call, GPU bound)
                generation_results = self.generate(prompts)

                # 2. If a previous sandbox execution is running, wait for it before
                #    starting the next batch. This ensures we don't pile up too many
                #    sandbox tasks and maintains order for logging/saving. Typically
                #    sandbox execution is much faster than LLM inference, so this should
                #    usually return immediately and not block generation.
                if prev_future:
                    batch_results = prev_future.result()
                    task_results.extend(batch_results)
                    self._save_results(batch_results, checkpoint)

                # 3. Process current generation in background (Sandbox is CPU bound)
                prev_future = executor.submit(
                    self._process_batch,
                    batch=batch,
                    generation_results=generation_results,
                    dataset_name=dataset_name,
                )

            # Wait for the last batch to finish sandbox execution.
            if prev_future:
                batch_results = prev_future.result()
                task_results.extend(batch_results)
                self._save_results(batch_results, checkpoint)

        metrics = self._calculate_metrics(
            dataset_name=dataset_name, checkpoint=checkpoint
        )
        return metrics, task_results

    def _process_batch(
        self,
        batch: dict[str, Any],
        generation_results: list[GenerationResult],
        dataset_name: str,
    ) -> list[TaskResult]:
        """Process a batch of LLM responses in the sandbox.

        This method is designed to be run in a background thread while the next
        batch is being generated by the model.

        Args:
            batch: Dictionary containing the batch data (a HuggingFace dataset object).
            generation_results: List of GenerationResult objects for the batch.
            dataset_name: Name of the dataset.
        """
        batch_results = []
        test_scripts = []

        for i, result in enumerate(generation_results):
            task_id = batch["task_id"][i]
            test_template = batch["test"][i]

            # TODO (Ryan Parker, 2026-04-24): For HumanEval, I need to consider how to handle
            # the extraction since the prompts expect the model to simply continue with the
            # function body rather than starting at the signature within a code block.
            # Some ideas:
            #   1. For HumanEval, concatenate the function signature with the model's completion.
            #   2. For both datasets, concatenate all Python code blocks from the completion as
            #      long as they are part of the model's final answer rather than the reasoning trace.
            extracted_code, correct_format = rewards.extract_code(
                result.completion, strategy="last"
            )

            # Format the code block to be executed in the sandbox. This includes the
            # model's generated code (the "solution") followed by the test cases, and it
            # ends with printing the pass/fail result as a JSON object for later parsing.
            if extracted_code:
                test_script = test_template.replace("{solution}", extracted_code)
                test_scripts.append(test_script)
            else:
                test_scripts.append(None)

            batch_results.append(
                TaskResult(
                    model_name=self.model_name,
                    model_id=self.model_id,
                    dataset_name=dataset_name,
                    task_id=task_id,
                    # "timestamp" is created automatically when this is initialized.
                    # "passed" and "pass_rate" are determined below.
                    correct_format=correct_format,
                    extraction_success=extracted_code is not None,
                    generation_time=result.generation_time,
                    # "sandbox_execution_time" is set below.
                    token_count=result.token_count,
                    throughput=result.throughput,
                    # "error_message" is set below.
                    prompt=batch["prompt"][i],
                    completion=result.completion,
                    extracted_code=extracted_code,
                )
            )

        # Execute test scripts in batch for all samples with extracted code.
        valid_indices = [i for i, s in enumerate(test_scripts) if s is not None]
        valid_scripts = [test_scripts[i] for i in valid_indices]

        if valid_scripts:
            exec_results = self.sandbox.run(valid_scripts)
            for i, exec_res in zip(valid_indices, exec_results):
                batch_results[i].sandbox_execution_time = exec_res.duration_sec

                outcome = self._safe_parse_json_result(exec_res.stdout)
                # For MBPP, stdout is a list of booleans
                # For HumanEval, stdout is a dict with a 'passed' key
                if isinstance(outcome, list):
                    batch_results[i].passed = all(outcome)
                    batch_results[i].pass_rate = rewards.pass_rate(outcome)
                elif isinstance(outcome, dict):
                    batch_results[i].passed = outcome.get("passed", False)
                else:
                    batch_results[i].passed = False

                if exec_res.failed:
                    batch_results[
                        i
                    ].error_message = f"Sandbox failed: {exec_res.errors}"

        return batch_results

    def _safe_parse_json_result(self, stdout: str | None) -> dict | list | None:
        """Safely parse the last line of JSON from the sandbox output.

        In `lfm_coder/datasets/eval_data.py`, the `test` column is modified to print
        a JSON object with test results (pass/fail) as the last line of stdout.
        """
        if not stdout or not stdout.strip():
            return None
        try:
            last_line = stdout.strip().split("\n")[-1]
            return json.loads(last_line)
        except (json.JSONDecodeError, IndexError):
            return None

    def _save_results(self, results: list[TaskResult], checkpoint: Checkpoint):
        """Append results to a JSONL file."""
        with checkpoint.checkpoint_file.open(mode="a", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(asdict(res)) + "\n")

    def _load_results(self, path: Path) -> list[TaskResult]:
        """Load results from a JSONL file."""
        results = []
        if not path.exists():
            return results

        with path.open(mode="r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                results.append(TaskResult(**data))
        return results

    def _calculate_metrics(
        self, dataset_name: str, checkpoint: Checkpoint
    ) -> DatasetMetrics:
        """
        Calculate metrics for a dataset.

        Filters out empty responses when calculating metrics to accommodate
        resumed checkpoints.

        Args:
            dataset_name: Name of the dataset.
            checkpoint: The checkpoint object for the dataset.

        Returns:
            DatasetMetrics object.
        """
        results = list(
            filter(
                lambda x: (
                    x.dataset_name == dataset_name
                    and x.token_count
                    and x.token_count > 0
                ),
                self._load_results(checkpoint.checkpoint_file),
            )
        )
        completed_tasks = len(results)
        if completed_tasks == 0:
            return DatasetMetrics(dataset_name=dataset_name)

        passed_count = sum(1 for r in results if r.passed)
        correct_format_count = sum(1 for r in results if r.correct_format)
        successful_extraction_count = sum(1 for r in results if r.extraction_success)
        total_exec_time = sum(r.sandbox_execution_time for r in results)
        total_tokens = sum(r.token_count for r in results)

        return DatasetMetrics(
            dataset_name=dataset_name,
            pass_rate=passed_count / completed_tasks,
            format_success_rate=correct_format_count / completed_tasks,
            extraction_success_rate=successful_extraction_count / completed_tasks,
            avg_sandbox_execution_time=total_exec_time / completed_tasks,
            avg_tokens=total_tokens / completed_tasks,
            total_tokens=total_tokens,
            completed_tasks=completed_tasks,
        )
