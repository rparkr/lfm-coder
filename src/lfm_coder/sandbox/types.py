"""Types for sandbox execution."""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from pydantic_monty import MemoryFile


class SandboxType(StrEnum):
    """Represent the type of sandbox used for code execution."""

    DOCKER = "docker"
    MONTY = "monty"
    CONTAINER_POOL = "container_pool"
    AUTO = "auto"


class SandboxError(Exception):
    """Base class for exceptions raised during sandbox execution."""

    def __init__(self, message: str, traceback: str | None = None):
        super().__init__(message)
        self.message = message
        self.traceback = traceback


class SandboxTimeoutError(SandboxError):
    """Raised when code execution exceeds the time limit."""

    pass


class SandboxMemoryError(SandboxError):
    """Raised when code execution exceeds the memory limit."""

    pass


class SandboxSyntaxError(SandboxError):
    """Raised when the code has syntax errors."""

    pass


class SandboxRuntimeError(SandboxError):
    """Raised when an error occurs during code execution."""

    pass


class SandboxNotSupportedError(SandboxError):
    """Raised when code uses features not supported by the sandbox (e.g., Monty)."""

    pass


@dataclass
class SandboxConfig:
    """Configuration for the sandbox environment.

    Attributes:
        max_duration_sec: Maximum time to run the code in seconds.
            Defaults to 10.0.
        max_memory_mb: Maximum memory (in MB) to allow for the sandbox.
            Defaults to 64.
        max_cpus: Maximum number of CPUs to use.
            Applies to `DockerSandbox` only.
        disable_network: Whether to disable network access.
            Applies to `DockerSandbox` only.
        use_cache: Whether to use a cache on the host machine for downloaded dependencies.
            Applies to `DockerSandbox` only.
        image_name: Name of the Docker image to use.
            Applies to `DockerSandbox` only.
        max_allocations: Maximum number of heap allocations.
            Applies to `MontySandbox` only.
        max_recursion_depth: Maximum depth of recursion allowed for sandboxed code.
            Applies to `MontySandbox` only.
    """

    max_duration_sec: float | None = 10.0
    max_memory_mb: int | None = 64

    # Docker settings only
    max_cpus: float | None = 1.0
    disable_network: bool | None = True
    use_cache: bool | None = False
    image_name: str | None = None

    # Monty settings only
    max_allocations: int | None = None
    max_recursion_depth: int | None = 1_000


@dataclass
class SandboxInput:
    """Input for sandboxed code execution.

    Attributes:
        code: The code to execute in the sandbox.
        env_vars: The environment variables to make available in the sandbox.
        input_files: The input files to make available in the sandbox. All files will
            be relative to the sandbox's working directory, which is `/sandbox`.
        dependencies: The PyPI packages to install in the sandbox.
            Applies to `DockerSandbox` only.
        external_functions: A dictionary of external functions to make available in the sandbox.
            Applies to `MontySandbox` only.
        inputs: A dictionary of input variables to make available in the sandbox. These
            are name-value pairs like {"x": 1, "y": 2, "z": [1, 2, 3]}.
            Applies to `MontySandbox` only.
        type_check: Whether to perform type checking on the code.
            Applies to `MontySandbox` only.
        type_check_stubs: The type stubs to use for type checking code and external
            functions.
            Applies to `MontySandbox` only.
        dataclass_registry: A list of dataclasses to register with the type checker.
            Applies to `MontySandbox` only.
    """

    code: str
    env_vars: dict[str, str] | None = None
    input_files: list[Path] | None = None

    # Docker-specific
    dependencies: list[str] | None = None

    # Monty-specific
    external_functions: dict[str, Callable] | None = None
    inputs: dict[str, Any] | None = None
    type_check: bool = False
    type_check_stubs: str | None = None
    dataclass_registry: list[type[Any]] | None = None


@dataclass
class SandboxExecution:
    """Results of code execution in the sandbox.

    Attributes:
        sandbox_type: The type of sandbox used for code execution.
        stdout: The standard output (e.g., print statements) during code execution.
        stderr: The standard error during code execution.
        success: Whether the code execution was successful.
        failed: Whether the code execution failed (i.e., had a non-zero exit code,
            timed out, hit memory limit, or had runtime errors).
        errors: A list of errors that occurred during code execution.
        exception_info: Information about the exception if the execution failed.
            Only present when `failed` is True.
        files: A list of MemoryFile objects representing files created or modified
            during execution. Applies to `MontySandbox` only.
        is_valid_python: Whether the code is valid Python syntax.
        exit_code: The exit code of the code execution.
        start_time: Timestamp in UTC when execution started.
        end_time: Timestamp in UTC when execution completed.
        duration_sec: How long the code execution took in seconds.
        timed_out: Whether the code execution timed out.
        memory_limit_hit: Whether the code exceeded the memory limit.
        inputs: The inputs to the code execution.
        result: The final result of the code execution, if any.
    """

    sandbox_type: SandboxType
    stdout: str
    stderr: str
    errors: list[SandboxError] | None = None
    files: list[MemoryFile] | None = None
    is_valid_python: bool = True
    exit_code: int = 0
    start_time: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    end_time: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    duration_sec: float = 0.0
    timed_out: bool = False
    memory_limit_hit: bool = False
    inputs: SandboxInput | None = None
    result: Any | None = None

    @property
    def success(self) -> bool:
        """Return True if the code execution was successful."""
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.memory_limit_hit
            and not self.errors
            and self.is_valid_python
        )

    @property
    def failed(self) -> bool:
        """Return True if the code execution failed."""
        return not self.success
