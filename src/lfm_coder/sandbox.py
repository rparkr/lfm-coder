"""Python sandbox module using the Monty library.

This module provides a secure way to execute Python code in a sandboxed
environment, capturing the execution results including final output and
print statements during code execution.
"""

from dataclasses import dataclass
from typing import Any

import pydantic_monty


@dataclass
class SandboxOutput:
    """A class to represent the output of the PythonSandbox execution.

    Attributes:
        output: The result of the last expression in the executed code.
        print_statements: A list of captured print statements during execution.
    """

    output: Any
    print_statements: list[str]


class PythonSandbox:
    """A class for securely running Python code using the Monty library.

    This class provides a simple interface to execute Python code in a sandboxed
    environment, capturing the output, stdout, and stderr. The Monty library ensures
    that the code cannot access the host filesystem, environment variables, or network
    unless explicitly allowed through external functions.

    Attributes:
        code: The Python code to be executed.
        inputs: List of input variable names expected by the code.
        max_memory_mb: Maximum memory limit in megabytes.
        max_duration_secs: Maximum execution time in seconds.
        max_allocations: Maximum number of heap allocations.
        resource_limits: ResourceLimits configuration for Monty.
        monty: The underlying Monty interpreter instance.
    """

    def __init__(
        self,
        code: str,
        inputs: list[str] | None = None,
        max_memory_mb: float | None = 512,
        max_duration_secs: float | None = 30.0,
        max_allocations: int | None = None,
    ) -> None:
        """Initialize the PythonSandbox.

        Args:
            code: The Python code to execute as a string.
            inputs: Optional list of input variable names that the code expects.
                Defaults to an empty list.
            max_memory_mb: Optional maximum memory limit in megabytes.
                Defaults to 512 MB. Set to None for no memory limit.
            max_duration_secs: Optional maximum execution time in seconds.
                Defaults to 30 seconds. Set to None for no time limit.
            max_allocations: Optional maximum number of heap allocations.
        """
        self.code = code
        self.inputs = inputs
        self.max_memory_mb = max_memory_mb
        self.max_duration_secs = max_duration_secs
        self.max_allocations = max_allocations
        self.print_statements = []

        resource_limit_mapping = {
            "max_memory": (
                int(max_memory_mb * 1024 * 1024) if max_memory_mb is not None else None
            ),
            "max_duration_secs": max_duration_secs,
            "max_allocations": max_allocations,
            "max_recusion_depth": 1000,
        }
        self.resource_limits = pydantic_monty.ResourceLimits(
            **{k: v for k, v in resource_limit_mapping.items() if v is not None}
        )

        self.monty = pydantic_monty.Monty(code, inputs=self.inputs)

    def __repr__(self) -> str:
        return (
            f"PythonSandbox(code={self.code!r}, inputs={self.inputs!r}, "
            f"max_memory_mb={self.max_memory_mb!r}, "
            f"max_duration_secs={self.max_duration_secs!r}, "
            f"max_allocations={self.max_allocations!r})"
        )

    def _capture_print(self, output_type: str, s: str) -> None:
        """Capture print statements during code execution.

        This method is used as a callback to capture print statements from the
        Monty interpreter.

        Args:
            output_type: The type of output ('stdout').
            s: The string output from a print statement.
        """
        if output_type == "stdout" and s.strip():
            self.print_statements.append(s)

    def run(self, **kwargs: Any) -> SandboxOutput:
        """Execute the Python code with the provided inputs.

        Captures stdout and stderr during execution using redirection.

        Args:
            **kwargs: Keyword arguments providing values for the input variables
                specified in the inputs list.

        Returns:
            A SandboxOutput instance containing:
            - 'output': The result of the last expression in the code.
            - 'print_statements': A list of captured print statements during execution.

        Raises:
            ValueError: If the provided inputs do not match the expected inputs.
            MontyRuntimeError: If there is a runtime error during code execution,
                Monty will raise an exception with details about the error.
        """
        if set(kwargs.keys()) != set(self.inputs or []):
            raise ValueError(
                f"Input mismatch: expected {self.inputs}, got {list(kwargs.keys())}"
            )

        # Capture output from only the current execution
        self.print_statements.clear()

        output = self.monty.run(
            inputs=kwargs if kwargs else None,
            limits=self.resource_limits,
            print_callback=self._capture_print,
            os=pydantic_monty.OSAccess(),
        )

        return SandboxOutput(output=output, print_statements=self.print_statements)
