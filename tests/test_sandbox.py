"""Unit tests for the Sandbox class (unified interface)."""

import textwrap

from lfm_coder.sandbox import Sandbox, SandboxExecution, SandboxType


class TestSandbox:
    """Test cases for the unified Sandbox class with fallback logic."""

    def test_sandbox_monty_auto(self):
        """Test that simple code runs in Monty by default."""
        sandbox = Sandbox()
        code = "print('Hello from Monty')\n1 + 2"
        result = sandbox.run(code)

        assert isinstance(result, SandboxExecution)
        assert result.sandbox_type == SandboxType.MONTY
        # Monty now appends repr(result) to stdout
        assert "Hello from Monty\n3" in result.stdout
        assert result.success is True

    def test_sandbox_fallback_to_docker_classes(self):
        """Test fallback to Docker for classes (unsupported by Monty)."""
        sandbox = Sandbox()
        code = textwrap.dedent("""
            class MyClass:
                def greet(self):
                    return "Hello from Docker"

            obj = MyClass()
            print(obj.greet())
        """)
        result = sandbox.run(code)

        assert isinstance(result, SandboxExecution)
        assert result.sandbox_type == SandboxType.DOCKER
        assert "Hello from Docker" in result.stdout
        assert result.success is True

    def test_sandbox_fallback_to_docker_dependencies(self):
        """Test fallback to Docker for external dependencies."""
        # Enable network so uv can fetch dependencies if needed
        sandbox = Sandbox(disable_network=False)
        code = textwrap.dedent("""
            import json
            import datetime
            # This should still be Monty compatible
            print("json and datetime are fine")
        """)
        result = sandbox.run(code)
        assert result.sandbox_type == SandboxType.MONTY

        code_with_dep = textwrap.dedent("""
            import numpy 
            print("numpy imported")
        """)
        # numpy is not in Monty's allowed modules
        result_dep = sandbox.run(code_with_dep)
        assert result_dep.sandbox_type == SandboxType.DOCKER

    def test_sandbox_batch(self):
        """Test batch execution with mixed compatibility."""
        sandbox = Sandbox()
        code_list = [
            "print('Task 1 - Monty')",
            "class A: pass\nprint('Task 2 - Docker')",
        ]
        results = sandbox.run(code_list)

        assert isinstance(results, list)
        assert len(results) == 2

        # Should be Monty
        assert results[0].sandbox_type == SandboxType.MONTY
        assert "Task 1 - Monty" in results[0].stdout
        assert results[0].result is None

        # Should be Docker
        assert results[1].sandbox_type == SandboxType.DOCKER
        assert "Task 2 - Docker" in results[1].stdout
        assert results[1].result is None

    def test_explicit_sandbox_type(self):
        """Test that explicit sandbox_type override works."""
        sandbox = Sandbox()
        code = "1 + 1"

        # Force Docker even if Monty compatible
        result = sandbox.run(code, sandbox_type=SandboxType.DOCKER)
        assert result.sandbox_type == SandboxType.DOCKER

        # Force Monty
        result_monty = sandbox.run(code, sandbox_type=SandboxType.MONTY)
        assert result_monty.sandbox_type == SandboxType.MONTY
        assert result_monty.result == 2
        assert result_monty.stdout == "2"

    def test_resource_override(self):
        """Test override of resource limits in run()."""
        sandbox = Sandbox(max_duration_sec=30)
        code = "import time; time.sleep(2)"

        # Override with very short timeout
        result = sandbox.run(code, max_duration_sec=0.1)
        assert result.timed_out is True
        assert result.exit_code == 124

    def test_overflow_prevention(self):
        """Test that the Sandbox prevents the 65536 column overflow in Monty by falling back to Docker."""
        sandbox = Sandbox()
        # Line length > 65536
        long_str = "a" * 70000
        code = f's = "{long_str}"\nprint(len(s))'

        result = sandbox.run(code)
        assert result.success is True
        assert result.stdout == "70000"
        assert result.sandbox_type == SandboxType.DOCKER

    def test_external_functions_docker_fallback(self):
        """Test that external_functions work when code falls back to Docker."""

        def my_helper(x: int) -> int:
            return x * 3

        # Class code falls back to Docker, but external_functions should still work
        code = textwrap.dedent("""
            class Calculator:
                def __init__(self):
                    self.value = 0

                def compute(self, x):
                    return my_helper(x)

            calc = Calculator()
            print(calc.compute(7))
        """)
        sandbox = Sandbox()
        result = sandbox.run(code, external_functions={"my_helper": my_helper})

        assert result.success is True
        assert "21" in result.stdout
        # Should have fallen back to Docker due to class usage
        assert result.sandbox_type == SandboxType.DOCKER

    def test_external_functions_monty(self):
        """Test that external_functions work in Monty when code is Monty-compatible."""

        def double(x: int) -> int:
            return x * 2

        # Simple code that runs in Monty
        code = "print(double(10))"
        sandbox = Sandbox()
        result = sandbox.run(code, external_functions={"double": double})

        assert result.success is True
        assert "20" in result.stdout
        assert result.sandbox_type == SandboxType.MONTY

    def test_external_functions_batch_docker(self):
        """Test external_functions in batch execution that falls back to Docker."""

        def transform(x: int) -> int:
            return x + 100

        code_list = [
            "print(transform(1))",
            "class Foo: pass\nprint(transform(2))",  # Second one uses class, falls to Docker
        ]
        sandbox = Sandbox()
        results = sandbox.run(code_list, external_functions={"transform": transform})

        assert len(results) == 2
        # First should be Monty
        assert results[0].sandbox_type == SandboxType.MONTY
        assert "101" in results[0].stdout
        # Second should be Docker (due to class)
        assert results[1].sandbox_type == SandboxType.DOCKER
        assert "102" in results[1].stdout
