import pytest

from lfm_coder.sandbox import DockerSandbox, SandboxExecution


@pytest.fixture
def sandbox():
    return DockerSandbox(max_duration_sec=10, max_memory_mb=128)


@pytest.fixture
def sandbox_low_memory():
    return DockerSandbox(max_duration_sec=10, max_memory_mb=16)


def test_simple_execution(sandbox):
    code = "print('Hello World')"
    result = sandbox.run(code)
    assert isinstance(result, SandboxExecution)
    assert "Hello World" in result.stdout
    assert result.success is True
    assert not result.timed_out


def test_dependency_installation(sandbox):
    # Enable network for this test so uv can download tqdm
    # Note: This might be slow or fail if no internet
    sandbox_nw = DockerSandbox(disable_network=False, max_duration_sec=60)
    code = "import tqdm; print('tqdm imported')"
    result = sandbox_nw.run(code)
    # We check if it either succeeded or if it's a known environment issue
    if result.failed:
        if "network" in result.stderr.lower() or "timeout" in result.stderr.lower():
            pytest.skip("Network issue or timeout during dependency installation")

    assert result.success is True
    assert "tqdm imported" in result.stdout


def test_timeout(sandbox):
    # Set a very short timeout for this test
    code = "import time; time.sleep(5)"
    # Standardized API: override max_duration_sec in run()
    result = sandbox.run(code, max_duration_sec=1)
    assert result.failed is True
    assert result.timed_out is True
    assert result.exit_code == 124


def test_memory_limit(sandbox_low_memory):
    # Allocate more than 16MB
    code = "x = ' ' * (1024 * 1024 * 32)"  # 32 MB string
    result = sandbox_low_memory.run(code)
    # Check if memory limit was hit
    assert result.failed is True
    assert result.memory_limit_hit is True
    assert result.exit_code == 137


def test_syntax_error(sandbox):
    code = "if True: print('broken syntax'"  # Missing closing parenthesis
    result = sandbox.run(code)
    assert result.failed is True
    assert result.is_valid_python is False
    assert "SyntaxError" in result.stderr


def test_disable_network(sandbox):
    # Default is disable_network=True
    code = (
        "import urllib.request; urllib.request.urlopen('https://google.com', timeout=1)"
    )
    result = sandbox.run(code)
    assert result.failed is True
    # Should fail with a network error/timeout since network is 'none'
    assert any(
        x in result.stderr
        for x in ["URLError", "Temporary failure", "timeout", "Network is unreachable"]
    )


def test_run_batch(sandbox):
    code_list = [f"print({i} * {i})" for i in range(3)]
    results = sandbox.run(code_list)

    assert isinstance(results, list)
    assert len(results) == 3
    for i, res in enumerate(results):
        assert res.success is True
        assert str(i * i) in res.stdout


def test_input_files_mount(sandbox, tmp_path):
    # Create a dummy file
    test_file = tmp_path / "data.txt"
    test_file.write_text("hello from host")

    code = "with open('/sandbox/data.txt', 'r') as f: print(f.read())"
    result = sandbox.run(code, input_files=[test_file])

    assert result.success is True
    assert "hello from host" in result.stdout


def test_env_vars(sandbox):
    code = "import os; print(os.environ.get('MY_VAR'))"
    result = sandbox.run(code, env_vars={"MY_VAR": "secret_value"})

    assert result.success is True
    assert "secret_value" in result.stdout
