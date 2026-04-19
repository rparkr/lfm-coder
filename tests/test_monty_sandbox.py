"""Unit tests for the MontySandbox class."""

import textwrap

from lfm_coder.sandbox import MontySandbox, SandboxExecution


class TestMontySandbox:
    """Test cases for the MontySandbox class."""

    def test_basic_execution(self):
        """Test basic code execution without inputs."""
        sandbox = MontySandbox()
        result = sandbox.run("42")

        assert isinstance(result, SandboxExecution)
        assert result.result == 42
        # REPL behavior: result is appended to stdout
        assert result.stdout == "42"

    def test_code_with_print(self):
        """Test code execution with print statements."""
        sandbox = MontySandbox()
        result = sandbox.run('print("Hello, World!"); 42')

        assert result.result == 42
        assert result.stdout == "Hello, World!\n42"

    def test_code_with_multiple_prints(self):
        """Test code execution with multiple print statements."""
        sandbox = MontySandbox()
        result = sandbox.run('print("First"); print("Second"); 99')

        assert result.result == 99
        assert result.stdout == "First\nSecond\n99"

    def test_execution_with_inputs(self):
        """Test code execution with input variables."""
        sandbox = MontySandbox()
        result = sandbox.run("x + y", x=5, y=10)

        assert result.result == 15
        assert result.stdout == "15"

    def test_execution_with_inputs_and_prints(self):
        """Test code execution with inputs and print statements."""
        sandbox = MontySandbox()
        result = sandbox.run('print(f"Sum: {x + y}"); x * y', x=3, y=4)

        assert result.result == 12
        assert result.stdout == "Sum: 7\n12"

    def test_memory_limit(self):
        """Test memory limit enforcement."""
        # Create a sandbox with very low memory limit (2 MB)
        sandbox = MontySandbox(max_memory_mb=2)
        # Allocate a large list to trigger memory limit
        result = sandbox.run("[i for i in range(1000000)]")

        assert result.failed is True
        assert result.memory_limit_hit is True
        assert result.exit_code == 137

    def test_time_limit(self):
        """Test time limit enforcement."""
        sandbox = MontySandbox(max_duration_sec=0.1)
        result = sandbox.run("sum(range(1_000_000_000))")

        assert result.failed is True
        assert result.timed_out is True
        assert result.exit_code == 124

    def test_allocation_limit(self):
        """Test allocation limit enforcement."""
        sandbox = MontySandbox(max_allocations=2)
        result = sandbox.run("a=[1] * 100\nb=[2] * 100\nc=[3] * 100")

        assert result.failed is True
        assert result.exit_code == 137
        assert result.memory_limit_hit is True

    def test_multiple_resource_limits(self):
        """Test multiple resource limits together."""
        sandbox = MontySandbox(
            max_memory_mb=10,
            max_duration_sec=1,
            max_allocations=1000,
        )
        result = sandbox.run("print('Test'); 42")

        assert result.result == 42
        assert result.stdout == "Test\n42"

    def test_no_resource_limits(self):
        """Test execution without any resource limits."""
        sandbox = MontySandbox(max_duration_sec=None, max_memory_mb=None)
        result = sandbox.run("42")

        assert result.result == 42
        assert result.stdout == "42"

    def test_runtime_error_handling(self):
        """Test that runtime errors are properly captured."""
        sandbox = MontySandbox()
        result = sandbox.run("1 / 0")

        assert result.failed is True
        assert result.errors is not None
        assert len(result.errors) > 0
        assert "ZeroDivisionError" in result.errors[0].message
        assert result.exit_code == 1

    def test_syntax_error_handling(self):
        """Test that syntax errors are caught."""
        sandbox = MontySandbox()
        result = sandbox.run("invalid syntax here +++")

        assert result.failed is True
        assert result.is_valid_python is False

    def test_async_execution(self):
        """Test async code execution."""
        sandbox = MontySandbox()
        code = textwrap.dedent("""
            async def task():
                return 42
            await task()
        """)
        # run automatically detects async
        result = sandbox.run(code)
        assert result.result == 42
        assert result.stdout == "42"

    def test_batch_execution(self):
        """Test batch execution in parallel."""
        sandbox = MontySandbox()
        codes = ["1+1", "2+2", "3+3"]
        results = sandbox.run(codes)

        assert isinstance(results, list)
        assert len(results) == 3
        assert results[0].result == 2
        assert results[1].result == 4
        assert results[2].result == 6

    def test_file_access(self):
        """Test file creation and access in Monty."""
        sandbox = MontySandbox()
        code = textwrap.dedent("""
            from pathlib import Path
            Path("/sandbox/test.txt").write_text("hello monty")
        """)
        result = sandbox.run(code)
        assert result.success is True
        assert result.files is not None
        assert len(result.files) > 0
        # Check if file exists in result.files
        found = False
        for f in result.files:
            if f.name == "test.txt":
                # Monty stores as string if write_text is used
                assert f.content in ["hello monty", b"hello monty"]
                found = True
        assert found is True

    def test_code_returning_none(self):
        """Test code that explicitly returns None."""
        sandbox = MontySandbox()
        result = sandbox.run("None")

        assert result.result is None
        assert result.stdout == ""

    def test_string_operations(self):
        """Test string operations in sandbox."""
        sandbox = MontySandbox()
        result = sandbox.run('"hello" + " world"')

        assert result.result == "hello world"
        assert result.stdout == "'hello world'"

    def test_list_operations(self):
        """Test list operations in sandbox."""
        sandbox = MontySandbox()
        result = sandbox.run("[1, 2, 3] + [4, 5]")

        assert result.result == [1, 2, 3, 4, 5]
        assert result.stdout == "[1, 2, 3, 4, 5]"

    def test_dict_operations(self):
        """Test dictionary operations in sandbox."""
        sandbox = MontySandbox()
        result = sandbox.run("{'a': 1, 'b': 2}")

        assert result.result == {"a": 1, "b": 2}
        assert result.stdout == "{'a': 1, 'b': 2}"

    def test_function_definition_and_call(self):
        """Test function definition and calling."""
        sandbox = MontySandbox()
        result = sandbox.run(
            textwrap.dedent("""
                def add(a, b):
                    return a + b

                add(5, 3)
            """)
        )

        assert result.result == 8
        assert result.stdout == "8"

    def test_complex_expression(self):
        """Test complex expressions."""
        sandbox = MontySandbox()
        result = sandbox.run(
            textwrap.dedent("""
                x = 10
                y = 20
                z = x * y + 5
                print(f"Result: {z}")
                z
            """)
        )

        assert result.result == 205
        assert result.stdout == "Result: 205\n205"

    def test_input_with_complex_types(self):
        """Test inputs with complex data types."""
        sandbox = MontySandbox()
        result = sandbox.run(
            textwrap.dedent("""
                print(f"List length: {len(data)}")
                sum(data)
            """),
            data=[1, 2, 3, 4, 5],
        )

        assert result.result == 15
        assert result.stdout == "List length: 5\n15"

    def test_print_with_newlines(self):
        """Test print statements with newlines."""
        sandbox = MontySandbox()
        result = sandbox.run('print("Line 1\\nLine 2"); 42')

        assert result.result == 42
        assert result.stdout == "Line 1\nLine 2\n42"
