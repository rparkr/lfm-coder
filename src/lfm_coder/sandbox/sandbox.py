"""Unified sandbox that falls back to Docker if Monty fails.

This module provides a unified interface for code execution, attempting to use the
faster MontySandbox first and falling back to the more robust DockerSandbox if
necessary.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, overload

from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox.docker_sandbox import DockerSandbox
from lfm_coder.sandbox.monty_sandbox import MontySandbox
from lfm_coder.sandbox.types import (
    SandboxExecution,
    SandboxType,
)

logger = get_logger(__name__)

# Modules supported by Monty, as of 2026-04-03
MONTY_SUPPORTED_MODULES = {
    "asyncio",
    "dataclasses",
    "datetime",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "sys",
    "typing",
}


@dataclass
class MontyCompatibility:
    """Represent the compatibility of code with the Monty sandbox.

    Attributes:
        is_compatible: Whether the code is compatible with Monty.
        reason: The reason for incompatibility, if any.
    """

    is_compatible: bool
    reason: str | None = None


class Sandbox:
    """Unified sandbox that combines MontySandbox and DockerSandbox.

    Attempt to execute code using MontySandbox for speed, and fall back to
    DockerSandbox if Monty cannot run the code due to unsupported features
    or libraries.
    """

    def __init__(
        self,
        max_duration_sec: float = 10.0,
        max_memory_mb: int = 64,
        max_cpus: float = 1.0,
        disable_network: bool = True,
        use_cache: bool = False,
        image_name: str | None = None,
        sandbox_type: SandboxType | Literal["auto"] = SandboxType.AUTO,
    ) -> None:
        """Initialize the unified sandbox.

        Args:
            max_duration_sec: Maximum execution time in seconds. Defaults to 10.0.
            max_memory_mb: Maximum memory limit in megabytes. Defaults to 64 MB.
            max_cpus: Maximum number of CPUs to use (Docker only).
            disable_network: Whether to disable network access (Docker only).
            use_cache: Whether to use uv cache (Docker only).
            image_name: Name of the Docker image to use.
            sandbox_type: Default sandbox type ("auto", "monty", or "docker").
        """
        self.sandbox_type = SandboxType(sandbox_type)
        self.monty = MontySandbox(
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
        )
        self.docker = DockerSandbox(
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            max_cpus=max_cpus,
            disable_network=disable_network,
            use_cache=use_cache,
            image_name=image_name,
        )

    @property
    def max_duration_sec(self) -> float | None:
        return self.monty.max_duration_sec

    @max_duration_sec.setter
    def max_duration_sec(self, value: float | None) -> None:
        self.monty.max_duration_sec = value
        self.docker.max_duration_sec = value

    @property
    def max_memory_mb(self) -> int | None:
        return self.monty.max_memory_mb

    @max_memory_mb.setter
    def max_memory_mb(self, value: int | None) -> None:
        self.monty.max_memory_mb = value
        self.docker.max_memory_mb = value

    @property
    def max_cpus(self) -> float | None:
        return self.docker.max_cpus

    @max_cpus.setter
    def max_cpus(self, value: float | None) -> None:
        self.docker.max_cpus = value

    @property
    def disable_network(self) -> bool | None:
        return self.docker.disable_network

    @disable_network.setter
    def disable_network(self, value: bool | None) -> None:
        self.docker.disable_network = value

    @property
    def use_cache(self) -> bool | None:
        return self.docker.use_cache

    @use_cache.setter
    def use_cache(self, value: bool) -> None:
        self.docker.use_cache = value

    @property
    def image_name(self) -> str:
        return self.docker.image_name

    @image_name.setter
    def image_name(self, value: str) -> None:
        self.docker.image_name = value

    @overload
    def run(
        self,
        code: str,
        sandbox_type: SandboxType | Literal["auto"] | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Any] | None = None,
        max_workers: int | None = None,
        skip_compatibility_check: bool = False,
        **kwargs: Any,
    ) -> SandboxExecution: ...

    @overload
    def run(
        self,
        code: list[str],
        sandbox_type: SandboxType | Literal["auto"] | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Any] | None = None,
        max_workers: int | None = None,
        skip_compatibility_check: bool = False,
        **kwargs: Any,
    ) -> list[SandboxExecution]: ...

    def run(
        self,
        code: str | list[str],
        sandbox_type: SandboxType | Literal["auto"] | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Any] | None = None,
        max_workers: int | None = None,
        skip_compatibility_check: bool = False,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code using the best available sandbox.

        Args:
            code: The Python code to execute (string or list of strings).
            sandbox_type: Override the default sandbox type for this execution.
            max_duration_sec: Optional override for the maximum execution time.
            max_memory_mb: Optional override for the maximum memory limit.
            max_cpus: Optional override for the maximum number of CPUs (Docker only).
            input_files: Files to make available in the sandbox.
            env_vars: Environment variables for the sandbox.
            external_functions: Dictionary of functions callable from inside the sandbox.
                Primarily meant for Monty, but compatible with Docker as well.
            max_workers: Maximum number of concurrent workers for batch execution.
            skip_compatibility_check: Whether to skip the compatibility check for Monty.
                If sandbox_type is "auto" this will still fall back to Docker if Monty
                fails to execute the code.
            **kwargs: Extra input variables for the code (Monty only).

        Returns:
            A single SandboxExecution or a list of them if a batch was provided.
        """
        stype = SandboxType(sandbox_type or self.sandbox_type)

        if isinstance(code, list):
            return self._run_batch(
                code,
                sandbox_type=stype,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                max_workers=max_workers,
                skip_compatibility_check=skip_compatibility_check,
                **kwargs,
            )

        # Single execution logic
        if stype == SandboxType.DOCKER:
            return self.docker.run(
                code,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                max_workers=max_workers,
                external_functions=external_functions,
                **kwargs,
            )

        if stype == SandboxType.MONTY:
            return self.monty.run(
                code,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                max_workers=max_workers,
                **kwargs,
            )

        # "auto" logic
        if not skip_compatibility_check:
            compatibility = self._can_run_in_monty(code)
            if not compatibility.is_compatible:
                logger.debug(
                    f"Monty incompatible: {compatibility.reason}. Falling back to Docker."
                )
                return self.docker.run(
                    code,
                    max_duration_sec=max_duration_sec,
                    max_memory_mb=max_memory_mb,
                    max_cpus=max_cpus,
                    input_files=input_files,
                    env_vars=env_vars,
                    max_workers=max_workers,
                    external_functions=external_functions,
                    **kwargs,
                )

        logger.debug("Attempting execution with MontySandbox")
        result = self.monty.run(
            code,
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            max_workers=max_workers,
            **kwargs,
        )

        # Fallback if Monty failed but it's valid Python
        if result.failed and result.is_valid_python:
            logger.debug(
                f"Monty failed on valid Python code (Errors: {[err.message for err in (result.errors or [])]}). "
                "Falling back to Docker."
            )
            return self.docker.run(
                code,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                max_workers=max_workers,
                external_functions=external_functions,
                **kwargs,
            )

        return result

    def _run_batch(
        self,
        code_batch: list[str],
        sandbox_type: SandboxType,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Any] | None = None,
        max_workers: int | None = None,
        skip_compatibility_check: bool = False,
        **kwargs: Any,
    ) -> list[SandboxExecution]:
        """Execute a batch of code using the optimized underlying sandboxes.

        Logic:
        1. Partition the batch into Monty-compatible and Docker-only tasks.
        2. Execute Monty and Docker sub-batches in parallel using their respective pooling logic.
        3. Reassemble results into original order.
        4. In AUTO mode, retry any Monty failures in Docker if the code is valid Python.

        Args:
            code_batch: List of Python scripts to execute.
            sandbox_type: Unified sandbox type ("auto", "monty", or "docker").
            max_duration_sec: Maximum execution time per task.
            max_memory_mb: Maximum memory per task.
            max_cpus: Maximum CPUs per task (Docker only).
            input_files: Shared input files for all tasks.
            env_vars: Environment variables for the sandbox.
            external_functions: Dictionary of functions callable from Monty.
            max_workers: Maximum workers for orchestration.
            skip_compatibility_check: Whether to bypass Monty module/feature checks.
            **kwargs: Additional variables for Monty.

        Returns:
            List of SandboxExecution results.
        """
        if sandbox_type == SandboxType.DOCKER:
            results = self.docker.run(
                code=code_batch,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                max_workers=max_workers,
                external_functions=external_functions,
                **kwargs,
            )
            return results

        if sandbox_type == SandboxType.MONTY:
            results = self.monty.run(
                code=code_batch,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                max_workers=max_workers,
                **kwargs,
            )
            return results

        # Auto mode: Partition into Monty and Docker
        monty_indices = []
        docker_indices = []
        monty_batch = []
        docker_batch = []

        for i, code in enumerate(code_batch):
            if skip_compatibility_check:
                monty_indices.append(i)
                monty_batch.append(code)
            else:
                compatibility_check = self._can_run_in_monty(code)
                if compatibility_check.is_compatible:
                    monty_indices.append(i)
                    monty_batch.append(code)
                else:
                    docker_indices.append(i)
                    docker_batch.append(code)

        # Create result placeholders, to be replaced with Docker/Monty results
        final_results = [None] * len(code_batch)

        # Run Monty batch
        if monty_batch:
            monty_results = self.monty.run(
                monty_batch,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                max_workers=max_workers,
                **kwargs,
            )
            for idx, res in zip(monty_indices, monty_results):
                final_results[idx] = res

        # Run Docker batch
        if docker_batch:
            docker_results = self.docker.run(
                docker_batch,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                max_workers=max_workers,
                external_functions=external_functions,
                **kwargs,
            )
            for idx, res in zip(docker_indices, docker_results):
                final_results[idx] = res

        # Post-process Monty results for fallbacks
        fallback_indices = []
        fallback_batch = []
        for i, res in enumerate(final_results):
            if (
                res
                and res.sandbox_type == SandboxType.MONTY
                and res.failed
                and res.is_valid_python
            ):
                fallback_indices.append(i)
                fallback_batch.append(code_batch[i])

        if fallback_batch:
            logger.debug(f"Retrying {len(fallback_batch)} Monty failures in Docker")
            fallback_results = self.docker.run(
                code=fallback_batch,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                max_cpus=max_cpus,
                input_files=input_files,
                env_vars=env_vars,
                max_workers=max_workers,
                external_functions=external_functions,
                **kwargs,
            )
            for idx, res in zip(fallback_indices, fallback_results):
                final_results[idx] = res

        # The type casting is safe here because all None values are replaced
        # with SandboxExecution objects.
        return cast(list[SandboxExecution], final_results)

    def _can_run_in_monty(self, code: str) -> MontyCompatibility:
        """Check if the code is compatible with Monty."""

        # The Monty parser fails when column numbers exceed 65,535 (u16::MAX)
        if any(len(line) > 65_535 for line in code.split("\n")):
            return MontyCompatibility(
                is_compatible=False,
                reason="Code contains lines longer than 65,536 characters",
            )

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Let Monty handle the syntax error reporting during actual run
            return MontyCompatibility(is_compatible=True)

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = []
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        modules.append(alias.name.split(".")[0])
                else:
                    if node.module:
                        modules.append(node.module.split(".")[0])

                unsupported_modules = list(set(modules) - MONTY_SUPPORTED_MODULES)
                if unsupported_modules:
                    return MontyCompatibility(
                        is_compatible=False,
                        reason=f"Unsupported module(s): {', '.join(unsupported_modules)}",
                    )

            if isinstance(node, (ast.ClassDef, ast.Match, ast.Delete)):
                feature = "Class"
                if isinstance(node, ast.Match):
                    feature = "Match"
                elif isinstance(node, ast.Delete):
                    feature = "Delete"

                return MontyCompatibility(
                    is_compatible=False,
                    reason=f"Unsupported feature: {feature} declaration",
                )

        return MontyCompatibility(is_compatible=True)
