"""Container-based sandbox for Python code execution using the llm-sandbox library.

Supports concurrent execution for evaluating batches of LLM-generated code
simultaneously using a pool of containers.

Securely run LLM-generated code in a container using Docker/Podman and uv
with full support for external libraries and all Python features.

Requires that either Docker or Podman is installed and running on the host machine.
If using Podman, alias it to `docker` (e.g., add `alias docker=podman` to your
shell's runtime configuration file like ~/.bashrc or ~/.zshrc).

**Note**:
In most cases, you'll want to use `DockerSandbox` over `PooledDockerSandbox` since
`DockerSandbox` is typically 2-5x faster and is less resource intensive compared to
this pooled version.
"""

from __future__ import annotations

import datetime
import importlib.util
from importlib.resources import as_file, files
from typing import Any

from lfm_coder.logging_utils import get_logger
from lfm_coder.sandbox.types import (
    SandboxError,
    SandboxExecution,
    SandboxInput,
    SandboxRuntimeError,
    SandboxTimeoutError,
    SandboxType,
)
from lfm_coder.sandbox.utils import detect_dependencies, load_module_mapping

logger = get_logger(__name__)

# llm-sandbox is an optional dependency
_has_llm_sandbox = importlib.util.find_spec("llm_sandbox") is not None

if _has_llm_sandbox:
    from llm_sandbox import SandboxBackend
    from llm_sandbox.pool import PoolConfig, create_pool_manager
    from llm_sandbox.session import SandboxSession
else:
    # Placeholders for when llm_sandbox is not installed
    from typing import TYPE_CHECKING

    class SandboxBackend:
        PODMAN = "podman"
        DOCKER = "docker"

    if TYPE_CHECKING:
        import llm_sandbox

    PoolConfig = None  # type: ignore[assignment]
    create_pool_manager = None  # type: ignore[assignment]
    SandboxSession = None  # type: ignore[assignment]


class PooledDockerSandbox:
    """Pooled Docker/Podman containers for Python code execution using llm-sandbox."""

    IMAGE_NAME: str = "lfm-coder-sandbox"

    def __init__(
        self,
        backend: "llm_sandbox.SandboxBackend" = "podman",  # type: ignore
        max_duration_sec: float = 10.0,
        max_memory_mb: int = 64,
        disable_network: bool = True,
        image_name: str | None = None,
        max_pool_size: int = 10,
        min_pool_size: int = 2,
        max_container_uses: int | None = None,
        idle_timeout: float = 30.0,
        verbose: bool = False,
    ) -> None:
        """Initialize a pooled Docker/Podman sandbox."""
        if not _has_llm_sandbox:
            raise ImportError(
                "llm-sandbox is missing. To use PooledDockerSandbox, "
                "please install with `uv pip install 'lfm-coder[container-pools]'`."
            )
        # Convert string to SandboxBackend enum if needed
        if isinstance(backend, str):
            backend = SandboxBackend(backend)  # type: ignore
        self.max_duration_sec = max_duration_sec
        self.max_memory_mb = max_memory_mb
        self.disable_network = disable_network
        self.image_name = image_name or self.IMAGE_NAME

        self.dockerfile_path = files("lfm_coder.sandbox").joinpath("Dockerfile.sandbox")
        self.verbose = verbose
        self._module_mapping = None

        # Initialize pool manager.
        pool_kwargs = {
            "backend": backend,
            "config": PoolConfig(  # type: ignore
                max_pool_size=max_pool_size,
                min_pool_size=min_pool_size,
                max_container_uses=max_container_uses,
                idle_timeout=idle_timeout,
            ),
            "lang": "python",
            "runtime_configs": {"mem_limit": f"{self.max_memory_mb}m"},
        }
        if self.disable_network:
            pool_kwargs["runtime_configs"]["network_mode"] = "none"

        if self._image_exists():
            pool_kwargs["image"] = self.image_name
            self._pool = create_pool_manager(**pool_kwargs)  # type: ignore
        else:
            with as_file(self.dockerfile_path) as df_path:
                pool_kwargs["dockerfile"] = str(df_path)
                self._pool = create_pool_manager(**pool_kwargs)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"PooledDockerSandbox("
            f"max_duration_sec={self.max_duration_sec!r}, "
            f"max_memory_mb={self.max_memory_mb!r}, "
            f"disable_network={self.disable_network!r}, "
            f"image_name={self.image_name!r}, "
            f"max_pool_size={self._pool.config.max_pool_size!r}, "
            f"min_pool_size={self._pool.config.min_pool_size!r}"
            ")"
        )

    def _image_exists(self) -> bool:
        import subprocess

        result = subprocess.run(
            ["docker", "images", "--quiet", self.image_name],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(result.stdout.strip())

    @property
    def module_mapping(self) -> dict[str, str]:
        if self._module_mapping is None:
            self._module_mapping = load_module_mapping()
        return self._module_mapping

    def run(
        self,
        code: str | list[str],
        max_workers: int | None = None,
        max_duration_sec: float | None = None,
        max_memory_mb: int | None = None,
        **kwargs: Any,
    ) -> SandboxExecution | list[SandboxExecution]:
        """Execute Python code in pooled container session(s)."""
        duration = max_duration_sec or self.max_duration_sec

        if isinstance(code, list):
            import concurrent.futures

            workers = max_workers or len(code)
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        self._run_single,
                        c,
                        max_duration_sec=duration,
                    )
                    for c in code
                ]
                return [f.result() for f in futures]

        return self._run_single(code, max_duration_sec=duration)

    def _run_single(
        self,
        code: str,
        max_duration_sec: float | None = None,
    ) -> SandboxExecution:
        """Execute a single code snippet in a pooled container."""
        duration = max_duration_sec or self.max_duration_sec
        libs = detect_dependencies(code, self.module_mapping)
        start_time = datetime.datetime.now(datetime.timezone.utc)
        sandbox_input = SandboxInput(code=code, dependencies=libs)

        try:
            with SandboxSession(  # type: ignore
                pool=self._pool,
                lang="python",
                verbose=self.verbose,
                # skip_environment_setup=False,
            ) as session:
                if libs:
                    # Install packages system-wide, following the pattern in llm_sandbox.
                    # See: https://vndee.github.io/llm-sandbox/custom-images/?h=system#how-it-works-python-specific
                    cmd_out = session.execute_command(
                        command=f"uv pip install --system {' '.join(libs)}"
                    )
                    if not cmd_out.success():
                        raise RuntimeError(
                            f"Failed to install libraries: {cmd_out.stderr}"
                        )

                result = session.run(code=code, timeout=duration)

                end_time = datetime.datetime.now(datetime.timezone.utc)
                actual_duration = (end_time - start_time).total_seconds()

                timed_out = actual_duration >= duration or result.exit_code == 124

                errors = []
                if result.exit_code != 0:
                    if result.exit_code == 124:
                        errors.append(
                            SandboxTimeoutError(
                                f"Execution timed out after {duration} seconds"
                            )
                        )
                    else:
                        errors.append(SandboxRuntimeError(result.stderr))

                return SandboxExecution(
                    sandbox_type=SandboxType.CONTAINER_POOL,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    errors=errors if errors else None,
                    exit_code=result.exit_code,
                    start_time=start_time,
                    end_time=end_time,
                    duration_sec=actual_duration,
                    timed_out=timed_out,
                    memory_limit_hit=result.exit_code == 137,
                    inputs=sandbox_input,
                )
        except Exception as e:
            msg = str(e)
            logger.error(f"Execution error in PooledDockerSandbox: {msg}")
            end_time = datetime.datetime.now(datetime.timezone.utc)
            duration = (end_time - start_time).total_seconds()

            timed_out = "timeout" in msg.lower() or "timed out" in msg.lower()
            exit_code = 124 if timed_out else 1
            error_obj = SandboxTimeoutError(msg) if timed_out else SandboxError(msg)

            return SandboxExecution(
                sandbox_type=SandboxType.CONTAINER_POOL,
                stdout="",
                stderr=msg,
                errors=[error_obj],
                exit_code=exit_code,
                start_time=start_time,
                end_time=end_time,
                duration_sec=duration,
                timed_out=timed_out,
                memory_limit_hit=False,
                inputs=sandbox_input,
            )

    def close(self):
        """Shut down the pool manager and clean up all containers."""
        if hasattr(self, "_pool") and self._pool:
            self._pool.close()

    def __enter__(self) -> "PooledDockerSandbox":
        """Enter context manager for proper resource cleanup."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and ensure containers are cleaned up."""
        self.close()

    def __del__(self):
        self.close()
