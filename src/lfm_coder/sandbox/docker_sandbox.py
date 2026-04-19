"""Container-based sandbox for Python code execution. Supports concurrent execution
for evaluating batches of LLM-generated code simultaneously using a thread pool.

Securely run LLM-generated code in a container using Docker/Podman and uv
with full support for external libraries and all Python features.

Requires that either Docker or Podman is installed and running on the host machine.
If using Podman, alias it to `docker` (e.g., add `alias docker=podman` to your
shell's runtime configuration file like ~/.bashrc or ~/.zshrc).

**Note**:
This is typically 2-5x faster than `PooledDockerSandbox` and is less resource intensive.

It is not quite as fast as `MontySandbox` because it has to start a container for each
execution, but it supports external libraries and all Python features so it should
generally be preferred over `MontySandbox` unless you know the code you are running
is supported by Monty.
"""

import concurrent.futures
import datetime
import os
import subprocess
import tempfile
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any, cast, overload

from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox.types import (
    SandboxConfig,
    SandboxError,
    SandboxExecution,
    SandboxInput,
    SandboxMemoryError,
    SandboxRuntimeError,
    SandboxTimeoutError,
    SandboxType,
)
from lfm_coder.sandbox.utils import detect_dependencies, load_module_mapping

logger = get_logger(__name__)


class DockerSandbox:
    """Security-focused Python execution sandbox using Docker/Podman and uv.

    This implementation uses ephemeral containers to ensure a clean slate for each
    execution and strictly limits resources (CPU, Memory, Network).
    """

    IMAGE_NAME: str = "lfm-coder-sandbox"

    def __init__(
        self,
        max_duration_sec: float = 10.0,
        max_memory_mb: int | None = 64,
        max_cpus: float | None = 1.0,
        disable_network: bool | None = True,
        use_cache: bool | None = False,
        image_name: str | None = None,
    ) -> None:
        """Create a container-based sandbox based on Docker/Podman.

        Args:
            max_duration_sec: Maximum time to run the code in seconds. Defaults to 10s.
            max_memory_mb: Maximum memory to use in MB. Defaults to 64 MB.
            max_cpus: Maximum number of CPUs to use per container. Defaults to 1 CPU.
            disable_network: Whether to disable network access. Defaults to True.
            use_cache: Whether to keep a cache on the host machine for uv-installed
                dependencies. This can significantly speed up subsequent runs when
                the code being executed contains third-party libraries. Defaults to False.
            image_name: Name of the Docker image to use.
        """
        self.config = SandboxConfig(
            max_duration_sec=max_duration_sec,
            max_memory_mb=max_memory_mb,
            max_cpus=max_cpus,
            disable_network=disable_network,
            use_cache=use_cache,
            image_name=image_name or self.IMAGE_NAME,
        )

        resources = files("lfm_coder.sandbox")
        self.dockerfile_path = resources.joinpath("Dockerfile.sandbox")

        self._image_ready = False
        self._module_mapping = None

    @property
    def max_duration_sec(self) -> int | float | None:
        return self.config.max_duration_sec

    @max_duration_sec.setter
    def max_duration_sec(self, value: int | float | None) -> None:
        self.config.max_duration_sec = value

    @property
    def max_memory_mb(self) -> int | None:
        return self.config.max_memory_mb

    @max_memory_mb.setter
    def max_memory_mb(self, value: int | None) -> None:
        self.config.max_memory_mb = value

    @property
    def max_cpus(self) -> float | None:
        return self.config.max_cpus

    @max_cpus.setter
    def max_cpus(self, value: float | None) -> None:
        self.config.max_cpus = value

    @property
    def disable_network(self) -> bool | None:
        return self.config.disable_network

    @disable_network.setter
    def disable_network(self, value: bool | None) -> None:
        self.config.disable_network = value

    @property
    def use_cache(self) -> bool | None:
        return self.config.use_cache

    @use_cache.setter
    def use_cache(self, value: bool | None) -> None:
        self.config.use_cache = value

    @property
    def image_name(self) -> str:
        return self.config.image_name or self.IMAGE_NAME

    @image_name.setter
    def image_name(self, value: str) -> None:
        self.config.image_name = value

    def __repr__(self) -> str:
        return (
            f"DockerSandbox("
            f"max_duration_sec={self.max_duration_sec!r}, "
            f"max_memory_mb={self.max_memory_mb!r}, "
            f"max_cpus={self.max_cpus!r}, "
            f"disable_network={self.disable_network!r}, "
            f"use_cache={self.use_cache!r}, "
            f"image_name={self.image_name!r}"
            ")"
        )

    @property
    def module_mapping(self) -> dict[str, str]:
        if self._module_mapping is None:
            self._module_mapping = load_module_mapping()
        return self._module_mapping

    def _ensure_image(self) -> None:
        if self._image_ready:
            return

        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "--quiet", self.image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            self._image_ready = True
            return

        logger.info(f"Building Docker image {self.image_name}...")

        # Use as_file to ensure we have a physical path for the Docker build context.
        with as_file(self.dockerfile_path) as df_path:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--tag",
                    self.image_name,
                    "--file",
                    str(df_path),
                ],
                check=True,
                capture_output=True,
            )
        self._image_ready = True

    @overload
    def run(
        self,
        code: str,
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution: ...

    @overload
    def run(
        self,
        code: list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[SandboxExecution]: ...

    def run(
        self,
        code: str | list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code in ephemeral Docker containers.

        Args:
            code: The Python code to execute (string or list of strings).
            max_workers: Maximum number of concurrent workers for batch execution.
            max_duration_sec: Optional override for the maximum execution time.
            max_memory_mb: Optional override for the maximum memory limit.
            max_cpus: Optional override for the maximum number of CPUs to use.
            input_files: Files to make available in the sandbox (mounted read-only).
            env_vars: Environment variables for the sandbox.
            **kwargs: Extra input arguments (to align with MontySandbox API).

        Returns:
            A single SandboxExecution or a list of them if a batch was provided.
        """
        # Set resource limits for this run
        duration = max_duration_sec or self.max_duration_sec or 10
        memory = max_memory_mb or self.max_memory_mb or 64
        cpus = max_cpus or self.max_cpus or 1

        if isinstance(code, list):
            # Default cap: cpu_count / max_cpus
            cpu_count = os.cpu_count() or 4
            default_max = (cpu_count // cpus) or 1
            workers = cast(int, max_workers or default_max)

            if max_workers and max_workers > default_max:
                logger.warning(
                    f"max_workers ({max_workers}) exceeds recommended limit "
                    f"({default_max}) based on CPU count and CPU limit."
                )

            logger.debug(
                f"Running Docker batch with {len(code)} items and {workers} workers"
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self._run_single,
                        c,
                        max_duration_sec=duration,
                        max_memory_mb=memory,
                        max_cpus=cpus,
                        input_files=input_files,
                        env_vars=env_vars,
                    )
                    for c in code
                ]
                return [f.result() for f in futures]

        return self._run_single(
            code,
            max_duration_sec=duration,
            max_memory_mb=memory,
            max_cpus=cpus,
            input_files=input_files,
            env_vars=env_vars,
        )

    def _run_single(
        self,
        code: str,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        max_cpus: float | None = None,
        input_files: list[Path] | None = None,
        env_vars: dict[str, str] | None = None,
    ) -> SandboxExecution:
        """Execute a single Python script in a Docker container."""
        # Use supplied values or fall back to instance defaults
        duration = max_duration_sec or self.max_duration_sec or 10
        memory = max_memory_mb or self.max_memory_mb or 64
        cpus = max_cpus or self.max_cpus or 1

        self._ensure_image()

        dependencies = detect_dependencies(code, self.module_mapping)
        processed_code = self._add_script_metadata(code, dependencies)

        start_time = datetime.datetime.now(datetime.timezone.utc)
        sandbox_input = SandboxInput(
            code=code,
            input_files=input_files,
            env_vars=env_vars,
            dependencies=dependencies,
        )

        stdout = ""
        stderr = ""
        exit_code = -1
        timed_out = False
        memory_limit_hit = False
        error = None

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            tmp_dir = Path(tmp_dir_name)
            code_file = tmp_dir / "code.py"
            code_file.write_text(processed_code)

            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--memory",
                f"{memory}m",
                "--cpus",
                str(cpus),
                "--volume",
                f"{code_file.absolute()}:/sandbox/code.py:ro,Z",
            ]

            # Mount input files as read-only
            if input_files:
                for path in input_files:
                    if path.exists():
                        docker_cmd.extend(
                            ["--volume", f"{path.absolute()}:/sandbox/{path.name}:ro,Z"]
                        )
                    else:
                        logger.warning(
                            f"Input file {path} does not exist, skipping mount."
                        )

            if env_vars:
                for key, value in env_vars.items():
                    docker_cmd.extend(["--env", f"{key}={value}"])

            if self.use_cache:
                docker_cmd.extend(["--volume", "lfm-coder-uv-cache:/root/.cache/uv:z"])

            if self.disable_network:
                docker_cmd.extend(["--network", "none"])

            docker_cmd.extend(
                [self.image_name, "uv", "run", "--script", "/sandbox/code.py"]
            )

            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=duration,
                )
                stdout = result.stdout
                stderr = result.stderr
                exit_code = result.returncode

                # Check for OOM exit code (137)
                if exit_code == 137:
                    memory_limit_hit = True
                    error = SandboxMemoryError("Memory limit exceeded (OOM)")
                elif exit_code != 0:
                    error = SandboxRuntimeError(
                        f"Execution failed with exit code {exit_code}"
                    )

            except subprocess.TimeoutExpired as e:
                timed_out = True
                exit_code = 124
                stdout = (
                    e.stdout.decode()
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
                stderr = (
                    e.stderr.decode()
                    if isinstance(e.stderr, bytes)
                    else (e.stderr or "")
                )
                error = SandboxTimeoutError(
                    f"Execution timed out after {duration} seconds"
                )
            except Exception as e:
                logger.error(f"Error during container execution: {e}")
                exit_code = 1
                stderr += f"\nInternal Sandbox Error: {str(e)}"
                error = SandboxError(str(e))

        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Check for syntax error in stderr
        is_valid_python = "SyntaxError" not in stderr

        return SandboxExecution(
            sandbox_type=SandboxType.DOCKER,
            # Remove trailing newlines, if any
            stdout=(stdout or "").strip(),
            stderr=(stderr or "").strip(),
            errors=[error] if error else [],
            files=[],
            is_valid_python=is_valid_python,
            exit_code=exit_code,
            start_time=start_time,
            end_time=end_time,
            duration_sec=duration,
            timed_out=timed_out,
            memory_limit_hit=memory_limit_hit,
            inputs=sandbox_input,
        )

    def _add_script_metadata(self, code: str, dependencies: list[str]) -> str:
        """Add PEP 723 inline script metadata for dependencies."""
        if not dependencies:
            return code

        metadata = ["# /// script", "# dependencies = ["]
        for dep in dependencies:
            metadata.append(f'#   "{dep}",')
        metadata.append("# ]")
        metadata.append("# ///")

        return "\n".join(metadata) + "\n\n" + code
