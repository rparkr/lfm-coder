"""Types for evaluation."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from PIL.ImagePath import Path


@dataclass
class GenerationResult:
    """Result of a single generation from an LLM.

    Attributes:
        completion: The LLM's complete response.
        token_count: Number of tokens in the generated response.
        generation_time: Time in seconds to generate the response.
        throughput: Number of tokens generated per second.
    """

    completion: str
    token_count: int
    generation_time: float
    throughput: float


@dataclass
class TaskResult:
    """Result for a single evaluation task.

    Attributes:
        model_name: Name of the model evaluated.
        model_id: Name used to identify the model in the result logs.
        dataset_name: Name of the dataset the task belongs to.
        task_id: Unique identifier for the task.
        timestamp: When the task was evaluated.
        passed: Whether the extracted code passed the tests.
        pass_rate: The proportion of tests cases that passed (0.0 - 1.0).
        correct_format: Whether the response followed the expected format (e.g., had code blocks).
        extraction_success: Whether code was successfully extracted.
        sandbox_execution_time: Time taken to execute the code in the sandbox.
        generation_time: Time taken to generate the response.
        token_count: Number of tokens in the completion (if available).
        throughput: Number of tokens generated per second.
        error_message: Error message if execution or extraction failed.
        prompt: The prompt given to the model.
        completion: The model's full response.
        extracted_code: The code extracted from the model's response.
    """

    model_name: str
    model_id: str
    dataset_name: str
    task_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    passed: bool = False
    pass_rate: float | None = None
    correct_format: bool = False
    extraction_success: bool = False
    sandbox_execution_time: float = 0.0
    generation_time: float = 0.0
    token_count: int | None = None
    throughput: float | None = None
    error_message: str | None = None
    prompt: list[dict[str, str]] = field(default_factory=list)
    completion: str = ""
    extracted_code: str | None = None


@dataclass
class DatasetMetrics:
    """Aggregated metrics for a dataset.

    Attributes:
        dataset_name: Name of the dataset.
        pass_rate: Percentage of tasks that passed.
        format_success_rate: Percentage of responses that followed the expected format.
        extraction_success_rate: Percentage of tasks where code was extracted.
        avg_sandbox_execution_time: Average time taken for execution.
        total_tokens: Total tokens generated.
        avg_tokens: Average tokens per task.
        total_tasks: Total number of tasks in the dataset.
        completed_tasks: Number of tasks completed in this run.
    """

    dataset_name: str
    pass_rate: float = 0.0
    format_success_rate: float = 0.0
    extraction_success_rate: float = 0.0
    avg_sandbox_execution_time: float = 0.0
    total_tokens: int = 0
    avg_tokens: float = 0.0
    completed_tasks: int = 0


@dataclass
class EvaluationResult:
    """Overall result for an evaluation run.

    Attributes:
        timestamp: When the evaluation started.
        model_name: Name of the model evaluated.
        metrics: Dictionary mapping dataset names to DatasetMetrics.
        task_results: List of all TaskResult objects.
    """

    model_name: str
    timestamp: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    metrics: dict[str, DatasetMetrics] = field(default_factory=dict)
    task_results: list[TaskResult] = field(default_factory=list)


@dataclass
class Checkpoint:
    """
    Represents an evaluation checkpoint for a specific dataset.
    """

    model_name: str
    model_id: str
    checkpoint_file: Path
    dataset_name: str
    completed_task_ids: set[str]
    incomplete_task_ids: set[str]
    is_complete: bool
