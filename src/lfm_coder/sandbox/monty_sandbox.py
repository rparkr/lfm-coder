"""Python sandbox module using the Pydantic-Monty library.

This module provides a secure way to execute Python code in a sandboxed
environment, capturing the execution results including final result and
print statements.

It is faster than DockerSandbox but supports only a subset of Python:
- Basic Python syntax, including sync/async functions
- Some standard library modules: asyncio, dataclasses, datetime, json, math, os, pathlib,
  re, sys, typing
- No third party libraries (users must implement external functions that the sandbox
  calls out to)
- No classes (support should come soon)
- No match statements (support should come soon)
"""

import ast
import asyncio
import concurrent.futures
import datetime
import inspect
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, cast, overload

import pydantic_monty

from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox.types import (
    SandboxConfig,
    SandboxError,
    SandboxExecution,
    SandboxInput,
    SandboxMemoryError,
    SandboxNotSupportedError,
    SandboxRuntimeError,
    SandboxSyntaxError,
    SandboxTimeoutError,
    SandboxType,
)

logger = get_logger(__name__)


# Lazily initialize a single background event loop for efficiently running asynchronous
# code through Monty if called from the synchronous method.
_BACKGROUND_LOOP: asyncio.AbstractEventLoop | None = None


class _MontyLoopThread:
    """A persistent background thread running an asyncio event loop.

    This class bridges synchronous calls to the asynchronous Monty
    execution engine efficiently, without the overhead of creating and closing
    event loops for each execution.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self.loop.run_forever, name="MontyLoopThread", daemon=True
        )
        self.thread.start()


def _get_background_loop() -> asyncio.AbstractEventLoop:
    """Get the global background event loop, initializing it if necessary."""
    global _BACKGROUND_LOOP
    if _BACKGROUND_LOOP is None:
        _BACKGROUND_LOOP = _MontyLoopThread().loop
    return _BACKGROUND_LOOP


@dataclass
class _MontyCodeValidation:
    """Internal record of code validation for the Monty sandbox."""

    monty: pydantic_monty.Monty | None
    is_valid_python: bool
    errors: list[SandboxError] | None = None


class MontySandbox:
    """Securely run Python code using the Monty library.

    Provides a simple interface to execute Python code in a sandboxed
    environment, capturing the final result and any print statements.
    """

    def __init__(
        self,
        max_allocations: int | None = None,
        max_duration_sec: float | None = 10.0,
        max_memory_mb: int | None = 64,
        max_recursion_depth: int | None = 1_000,
    ) -> None:
        """Initialize a Monty sandbox for securely running code.

        Args:
            max_allocations: Optional maximum number of heap allocations.
            max_duration_sec: Maximum execution time in seconds.
            max_memory_mb: Maximum memory limit in megabytes.
            max_recursion_depth: Maximum depth of recursion allowed.
        """
        self.config = SandboxConfig(
            max_allocations=max_allocations,
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            max_recursion_depth=max_recursion_depth,
        )

    @property
    def max_allocations(self) -> int | None:
        return self.config.max_allocations

    @max_allocations.setter
    def max_allocations(self, value: int | None) -> None:
        self.config.max_allocations = value

    @property
    def max_duration_sec(self) -> float | None:
        return self.config.max_duration_sec

    @max_duration_sec.setter
    def max_duration_sec(self, value: float | None) -> None:
        self.config.max_duration_sec = value

    @property
    def max_memory_mb(self) -> int | None:
        return self.config.max_memory_mb

    @max_memory_mb.setter
    def max_memory_mb(self, value: int | None) -> None:
        self.config.max_memory_mb = value

    @property
    def max_recursion_depth(self) -> int | None:
        return self.config.max_recursion_depth

    @max_recursion_depth.setter
    def max_recursion_depth(self, value: int | None) -> None:
        self.config.max_recursion_depth = value

    def __repr__(self) -> str:
        return (
            f"MontySandbox("
            f"max_allocations={self.max_allocations!r}, "
            f"max_duration_sec={self.max_duration_sec!r}, "
            f"max_memory_mb={self.max_memory_mb!r}, "
            f"max_recursion_depth={self.max_recursion_depth!r}"
            ")"
        )

    @overload
    def run(
        self,
        code: str,
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution: ...

    @overload
    def run(
        self,
        code: list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> list[SandboxExecution]: ...

    def run(
        self,
        code: str | list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code in a Monty sandbox.

        Automatically handles both single code snippets and batches. Detects if
        asynchronous execution is required and bridges to an event loop if necessary.

        Args:
            code: The Python code to execute (string or list of strings).
            max_workers: Maximum number of concurrent workers for batch execution.
            max_duration_sec: Optional override for the maximum execution time.
            max_memory_mb: Optional override for the maximum memory limit.
            input_files: Files to make available in the sandbox.
            env_vars: Environment variables for the sandbox.
            external_functions: Dictionary of functions callable from inside the sandbox.
            type_check: Whether to perform type checking on the code prior to execution.
            type_check_stubs: Type stubs for type checking.
            dataclass_registry: Dataclasses to register with the type checker.
            **kwargs: Input variables to make available in the sandbox. Use when the code
                to execute contains references to variables that are not defined in the
                code itself.

        Returns:
            A single SandboxExecution or a list of them if a batch was provided.
        """
        if isinstance(code, list):
            return self._run_batch(
                self._run_sync_or_async,
                code,
                max_workers=max_workers,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
                dataclass_registry=dataclass_registry,
                **kwargs,
            )

        return self._run_sync_or_async(
            code,
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
            **kwargs,
        )

    def run_sync(
        self,
        code: str | list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code synchronously in a Monty sandbox.

        Args:
            code: The Python code to execute (string or list of strings).
            max_workers: Maximum number of concurrent workers for batch execution.
            max_duration_sec: Optional override for the maximum execution time.
            max_memory_mb: Optional override for the maximum memory limit.
            input_files: Files to make available in the sandbox.
            env_vars: Environment variables for the sandbox.
            external_functions: Dictionary of functions callable from inside the sandbox.
            type_check: Whether to perform type checking on the code prior to execution.
            type_check_stubs: Type stubs for type checking.
            dataclass_registry: Dataclasses to register with the type checker.
            **kwargs: Input variables to make available in the sandbox. Use when the code
                to execute contains references to variables that are not defined in the
                code itself.

        Returns:
            A single SandboxExecution or a list of them if a batch was provided.
        """
        if isinstance(code, list):
            return self._run_batch(
                self._run_single_sync,
                code,
                max_workers=max_workers,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
                dataclass_registry=dataclass_registry,
                **kwargs,
            )

        return self._run_single_sync(
            code,
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
            **kwargs,
        )

    async def run_async(
        self,
        code: str | list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code asynchronously in a Monty sandbox.

        Args:
            code: The Python code to execute (string or list of strings).
            max_workers: Maximum number of concurrent workers for batch execution.
            max_duration_sec: Optional override for the maximum execution time.
            max_memory_mb: Optional override for the maximum memory limit.
            input_files: Files to make available in the sandbox.
            env_vars: Environment variables for the sandbox.
            external_functions: Dictionary of functions callable from inside the sandbox.
            type_check: Whether to perform type checking on the code prior to execution.
            type_check_stubs: Type stubs for type checking.
            dataclass_registry: Dataclasses to register with the type checker.
            **kwargs: Input variables to make available in the sandbox. Use when the code
                to execute contains references to variables that are not defined in the
                code itself.

        Returns:
            A single SandboxExecution or a list of them if a batch was provided.
        """
        if isinstance(code, list):
            return await self._run_batch_async(
                self._run_single_async,
                code,
                max_workers=max_workers,
                max_duration_sec=max_duration_sec,
                max_memory_mb=max_memory_mb,
                input_files=input_files,
                env_vars=env_vars,
                external_functions=external_functions,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
                dataclass_registry=dataclass_registry,
                **kwargs,
            )

        return await self._run_single_async(
            code,
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
            **kwargs,
        )

    def _run_batch(
        self,
        func: Callable,
        code: list[str],
        max_workers: int | None = None,
        **kwargs: Any,
    ) -> list[SandboxExecution]:
        """Internal helper to execute code in batches."""
        workers = max_workers or os.cpu_count() or 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(func, c, **kwargs) for c in code]
            return [f.result() for f in futures]

    async def _run_batch_async(
        self,
        func: Callable,
        code: list[str],
        **kwargs: Any,
    ) -> list[SandboxExecution]:
        """Internal helper to execute code in batches asynchronously."""
        tasks = [func(c, **kwargs) for c in code]
        return list(await asyncio.gather(*tasks))

    def _run_sync_or_async(
        self,
        code: str,
        **kwargs: Any,
    ) -> SandboxExecution:
        """Internal helper to bridge sync calls to async if needed, by running the async version in a background thread."""
        external_functions = kwargs.get("external_functions")
        if self._requires_async(code, external_functions):
            loop = _get_background_loop()
            future = asyncio.run_coroutine_threadsafe(
                self._run_single_async(code, **kwargs),
                loop,
            )
            return future.result()
        else:
            return self._run_single_sync(code, **kwargs)

    def _run_single_sync(
        self,
        code: str,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution:
        """Run a single code item synchronously."""
        start_time = datetime.datetime.now(datetime.timezone.utc)
        sandbox_input = SandboxInput(
            code=code,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            inputs=kwargs,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
        )

        monty_code_validation = self._validate_code(
            code,
            inputs=kwargs,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
        )
        if monty_code_validation.errors:
            return self._create_result_for_failure(
                monty_code_validation, start_time, sandbox_input
            )

        os_access = self._setup_os_access(input_files, env_vars)
        print_statements = []

        def capture_print(output_type: str, s: str) -> None:
            """
            Capture print statements from the sandbox.
            As of 2026-04-03, Monty sets output_type to "stdout"
            """
            if output_type == "stdout" and s.strip():
                print_statements.append(s)

        exec_error = None
        result = None

        limits = self._get_monty_resource_limits(
            max_duration_sec=max_duration_sec, max_memory_mb=max_memory_mb
        )

        monty_runner = cast(pydantic_monty.Monty, monty_code_validation.monty)

        try:
            result = monty_runner.run(
                inputs=kwargs if kwargs else None,
                limits=limits,
                external_functions=external_functions,
                print_callback=capture_print,
                os=os_access,
            )
        # Catch "pyo3_runtime.PanicException" which are raised as BaseException
        except (Exception, BaseException) as e:
            exec_error = self._wrap_exception(e)

        if not exec_error and result is not None:
            print_statements.append(repr(result))

        end_time = datetime.datetime.now(datetime.timezone.utc)
        return self._create_execution_result(
            sandbox_input,
            result,
            print_statements,
            cast(list[pydantic_monty.MemoryFile], os_access.files),
            exec_error,
            start_time,
            end_time,
            monty_code_validation.is_valid_python,
        )

    async def _run_single_async(
        self,
        code: str,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        external_functions: dict[str, Callable] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution:
        """Core logic for a single asynchronous execution."""
        start_time = datetime.datetime.now(datetime.timezone.utc)
        sandbox_input = SandboxInput(
            code=code,
            input_files=input_files,
            env_vars=env_vars,
            external_functions=external_functions,
            inputs=kwargs,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
        )

        monty_code_validation = self._validate_code(
            code,
            inputs=kwargs,
            type_check=type_check,
            type_check_stubs=type_check_stubs,
            dataclass_registry=dataclass_registry,
        )
        if monty_code_validation.errors:
            return self._create_result_for_failure(
                monty_code_validation, start_time, sandbox_input
            )

        os_access = self._setup_os_access(input_files, env_vars)
        print_statements = []

        def capture_print(output_type: str, s: str) -> None:
            if output_type == "stdout" and s.strip():
                print_statements.append(s)

        exec_error = None
        result = None

        limits = self._get_monty_resource_limits(
            max_duration_sec=max_duration_sec, max_memory_mb=max_memory_mb
        )

        monty_runner = cast(pydantic_monty.Monty, monty_code_validation.monty)

        try:
            result = await monty_runner.run_async(
                inputs=kwargs if kwargs else None,
                limits=limits,
                external_functions=external_functions,
                print_callback=capture_print,
                os=os_access,
            )
        # Catch "pyo3_runtime.PanicException" which are raised as BaseException
        except (Exception, BaseException) as e:
            exec_error = self._wrap_exception(e)

        if not exec_error and result is not None:
            print_statements.append(repr(result))

        end_time = datetime.datetime.now(datetime.timezone.utc)
        return self._create_execution_result(
            sandbox_input,
            result,
            print_statements,
            cast(list[pydantic_monty.MemoryFile], os_access.files),
            exec_error,
            start_time,
            end_time,
            monty_code_validation.is_valid_python,
        )

    def _is_binary(self, path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with path.open(mode="rb") as f:
                chunk = f.read(1024)
                # Check for null bytes
                if b"\x00" in chunk:
                    return True
                return False
        except IOError:
            return False

    def _setup_os_access(
        self, input_files: list[Path] | None, env_vars: dict[str, str] | None
    ) -> pydantic_monty.OSAccess:
        """Prepare the OSAccess object with MemoryFiles."""
        memory_files = []
        if input_files:
            for path in input_files:
                try:
                    if self._is_binary(path):
                        content = path.read_bytes()
                    else:
                        content = path.read_text(encoding="utf-8")

                    memory_files.append(pydantic_monty.MemoryFile(path.name, content))
                except Exception as e:
                    logger.warning(f"Failed to read input file {path}: {e}")

        os_access = pydantic_monty.OSAccess(
            files=memory_files, environ=env_vars, root_dir="/sandbox"
        )
        # Ensure that the /sandbox directory exists
        try:
            os_access.path_mkdir(
                path=PurePosixPath("/sandbox"), parents=True, exist_ok=True
            )
        except Exception as e:
            logger.warning(f"Failed to create /sandbox directory: {e}")
        return os_access

    def _is_valid_python(self, code: str) -> bool:
        """Check if the code is valid Python."""

        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _validate_code(
        self,
        code: str,
        inputs: dict[str, Any] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
        dataclass_registry: list[type[Any]] | None = None,
    ) -> _MontyCodeValidation:
        """Parse the code with Monty and check for syntax errors."""
        input_names = list(inputs.keys()) if inputs else None
        try:
            monty = pydantic_monty.Monty(
                code,
                inputs=input_names,
                type_check=type_check,
                type_check_stubs=type_check_stubs,
                dataclass_registry=dataclass_registry,
            )
            return _MontyCodeValidation(monty=monty, is_valid_python=True, errors=None)
        except pydantic_monty.MontySyntaxError as e:
            return _MontyCodeValidation(
                monty=None,
                is_valid_python=self._is_valid_python(code),
                errors=[SandboxSyntaxError(str(e))],
            )
        # Catch "pyo3_runtime.PanicException" which are raised as BaseException
        except (Exception, BaseException) as e:
            return _MontyCodeValidation(
                monty=None,
                is_valid_python=self._is_valid_python(code),
                errors=[self._wrap_exception(e)],
            )

    def _wrap_exception(self, exc: Exception | BaseException) -> SandboxError:
        """Convert pydantic_monty exceptions to standard SandboxError."""
        msg = str(exc)
        traceback = ""
        if hasattr(exc, "traceback"):
            tb = getattr(exc, "traceback")
            traceback = tb() if callable(tb) else str(tb)

        if isinstance(exc, pydantic_monty.MontyRuntimeError):
            if "TimeoutError" in msg:
                return SandboxTimeoutError(msg, traceback=traceback)
            if "MemoryError" in msg:
                return SandboxMemoryError(msg, traceback=traceback)
            return SandboxRuntimeError(msg, traceback=traceback)

        if isinstance(exc, NotImplementedError):
            return SandboxNotSupportedError(msg, traceback=traceback)

        return SandboxError(msg, traceback=traceback)

    def _get_monty_resource_limits(
        self,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
    ) -> pydantic_monty.ResourceLimits:
        """Convert SandboxConfig or explicit overrides to pydantic_monty.ResourceLimits."""
        duration = (
            max_duration_sec
            if max_duration_sec is not None
            else self.config.max_duration_sec
        )
        memory = (
            max_memory_mb if max_memory_mb is not None else self.config.max_memory_mb
        )

        mapping = {
            "max_allocations": self.config.max_allocations,
            "max_duration_secs": duration,
            "max_memory": int(memory * 1024 * 1024) if memory is not None else None,
            "max_recursion_depth": self.config.max_recursion_depth,
        }
        return pydantic_monty.ResourceLimits(
            **{k: v for k, v in mapping.items() if v is not None}
        )

    def _create_execution_result(
        self,
        sandbox_input: SandboxInput,
        result: Any | None,
        stdout: list[str],
        memory_files: list[pydantic_monty.MemoryFile],
        error: SandboxError | None,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        is_valid_python: bool,
    ) -> SandboxExecution:
        """Assemble the final SandboxExecution object."""
        result_stdout = "\n".join(stdout)
        errors = [error] if error else []
        duration = (end_time - start_time).total_seconds()

        exit_code = 0
        if error:
            exit_code = 1
            if isinstance(error, SandboxTimeoutError):
                exit_code = 124
            elif isinstance(error, SandboxMemoryError):
                exit_code = 137

        return SandboxExecution(
            sandbox_type=SandboxType.MONTY,
            # Remove trailing newlines, if any
            stdout=result_stdout.strip(),
            stderr="",
            errors=errors,
            files=memory_files,
            is_valid_python=is_valid_python,
            exit_code=exit_code,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration,
            timed_out=isinstance(error, SandboxTimeoutError),
            memory_limit_hit=isinstance(error, SandboxMemoryError),
            inputs=sandbox_input,
            result=result,
        )

    def _create_result_for_failure(
        self,
        monty_code_validation: _MontyCodeValidation,
        start_time: datetime.datetime,
        sandbox_input: SandboxInput,
    ) -> SandboxExecution:
        """Create a SandboxExecution for validation failures."""
        end_time = datetime.datetime.now(datetime.timezone.utc)
        exit_code = 1
        if monty_code_validation.errors:
            error = monty_code_validation.errors[0]
            if isinstance(error, SandboxTimeoutError):
                exit_code = 124
            elif isinstance(error, SandboxMemoryError):
                exit_code = 137
        return SandboxExecution(
            sandbox_type=SandboxType.MONTY,
            stdout="",
            stderr="",
            errors=monty_code_validation.errors,
            files=None,
            is_valid_python=monty_code_validation.is_valid_python,
            exit_code=exit_code,
            start_time=start_time,
            end_time=end_time,
            duration_sec=0.0,
            inputs=sandbox_input,
            result=None,
        )

    def _requires_async(
        self, code: str, external_functions: dict[str, Callable] | None = None
    ) -> bool:
        """Check if the code or functions require async execution."""
        if external_functions:
            for func in external_functions.values():
                if inspect.iscoroutinefunction(func):
                    return True
        # Here's how that regex works: https://regex101.com/r/W94oNh/1
        return bool(re.search(r"^(\s*async def)|^([^#\"']*await)", code, re.MULTILINE))
