import pytest

pytest.importorskip("llm_sandbox")

from lfm_coder.sandbox import PooledDockerSandbox


@pytest.fixture(scope="module")
def sandbox():
    # Use a small pool for testing with sufficient memory for package installation
    with PooledDockerSandbox(
        min_pool_size=1,
        max_pool_size=2,
        disable_network=False,
        max_memory_mb=512,
    ) as sb:
        yield sb


def test_simple_execution(sandbox):
    code = "print(1 + 1)"
    result = sandbox.run(code)
    assert result.exit_code == 0
    assert result.stdout.strip() == "2"


def test_dependency_installation(sandbox):
    # 'tqdm' should be detected and installed
    code = "import tqdm\nprint(tqdm.__name__)"
    result = sandbox.run(code)
    assert result.exit_code == 0
    assert "tqdm" in result.stdout


def test_run_batch(sandbox):
    codes = [f"print({i} * {i})" for i in range(3)]
    results = sandbox.run(codes)

    assert len(results) == 3
    for i, res in enumerate(results):
        assert res.exit_code == 0
        assert str(i * i) in res.stdout


def test_timeout(sandbox):
    code = "import time; time.sleep(2)"
    result = sandbox.run(code, max_duration_sec=0.5)
    assert result.timed_out is True
    # Standard timeout exit code is 124
    assert result.exit_code == 124
