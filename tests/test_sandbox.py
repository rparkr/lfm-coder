"""Unit tests for the PythonSandbox class."""

import pytest
import pydantic_monty

from lfm_coder.sandbox import PythonSandbox, SandboxOutput


class TestPythonSandbox:
    """Test cases for the PythonSandbox class."""

    def test_basic_execution(self):
        """Test basic code execution without inputs."""
        sandbox = PythonSandbox("42")
        result = sandbox.run()

        assert isinstance(result, SandboxOutput)
        assert result.output == 42
        assert result.print_statements == []

    def test_code_with_print(self):
        """Test code execution with print statements."""
        sandbox = PythonSandbox('print("Hello, World!"); 42')
        result = sandbox.run()

        assert result.output == 42
        assert result.print_statements == ["Hello, World!"]

    def test_code_with_multiple_prints(self):
        """Test code execution with multiple print statements."""
        sandbox = PythonSandbox('print("First"); print("Second"); 99')
        result = sandbox.run()

        assert result.output == 99
        assert result.print_statements == ["First", "Second"]

    def test_execution_with_inputs(self):
        """Test code execution with input variables."""
        sandbox = PythonSandbox("x + y", inputs=["x", "y"])
        result = sandbox.run(x=5, y=10)

        assert result.output == 15
        assert result.print_statements == []

    def test_execution_with_inputs_and_prints(self):
        """Test code execution with inputs and print statements."""
        sandbox = PythonSandbox('print(f"Sum: {x + y}"); x * y', inputs=["x", "y"])
        result = sandbox.run(x=3, y=4)

        assert result.output == 12
        assert result.print_statements == ["Sum: 7"]

    def test_input_validation_missing_input(self):
        """Test that missing inputs raise ValueError."""
        sandbox = PythonSandbox("x + 1", inputs=["x"])

        with pytest.raises(ValueError, match="Input mismatch"):
            sandbox.run()

    def test_input_validation_extra_input(self):
        """Test that extra inputs raise ValueError."""
        sandbox = PythonSandbox("x + 1", inputs=["x"])

        with pytest.raises(ValueError, match="Input mismatch"):
            sandbox.run(x=5, y=10)

    def test_input_validation_wrong_input_names(self):
        """Test that wrong input names raise ValueError."""
        sandbox = PythonSandbox("x + 1", inputs=["x"])

        with pytest.raises(ValueError, match="Input mismatch"):
            sandbox.run(y=5)

    def test_memory_limit(self):
        """Test memory limit enforcement."""
        # Create a sandbox with very low memory limit (0.1 kB)
        sandbox = PythonSandbox("[i for i in range(100)]", max_memory_mb=0.0001)

        with pytest.raises(pydantic_monty.MontyRuntimeError):
            sandbox.run()

    def test_time_limit(self):
        """Test time limit enforcement."""
        # Create a computationally intensive task with short time limit
        sandbox = PythonSandbox("sum(range(10000000))", max_duration_secs=0.01)

        with pytest.raises(
            pydantic_monty.MontyRuntimeError, match="time limit exceeded"
        ):
            sandbox.run()

    def test_allocation_limit(self):
        """Test allocation limit enforcement."""
        # Create code that does many allocations with low limit
        sandbox = PythonSandbox(
            "a=[1] * 100\nb=[2] * 100\nc=[3] * 100", max_allocations=2
        )

        with pytest.raises(pydantic_monty.MontyError):
            sandbox.run()

    def test_multiple_resource_limits(self):
        """Test multiple resource limits together."""
        sandbox = PythonSandbox(
            "print('Test'); 42",
            max_memory_mb=10,
            max_duration_secs=1,
            max_allocations=1000,
        )
        result = sandbox.run()

        assert result.output == 42
        assert result.print_statements == ["Test"]

    def test_no_resource_limits(self):
        """Test execution without any resource limits."""
        sandbox = PythonSandbox("42")
        result = sandbox.run()

        assert result.output == 42
        assert result.print_statements == []

    def test_runtime_error_handling(self):
        """Test that runtime errors are properly raised."""
        sandbox = PythonSandbox("1 / 0")

        with pytest.raises(pydantic_monty.MontyRuntimeError):
            sandbox.run()

    def test_syntax_error_handling(self):
        """Test that syntax errors are caught during initialization."""
        with pytest.raises(pydantic_monty.MontySyntaxError):
            PythonSandbox("invalid syntax here +++")

    def test_empty_code(self):
        """Test execution of empty code."""
        sandbox = PythonSandbox("")
        result = sandbox.run()

        # Empty code should return None or similar
        assert result.output is None
        assert result.print_statements == []

    def test_code_returning_none(self):
        """Test code that explicitly returns None."""
        sandbox = PythonSandbox("None")
        result = sandbox.run()

        assert result.output is None
        assert result.print_statements == []

    def test_string_operations(self):
        """Test string operations in sandbox."""
        sandbox = PythonSandbox('"hello" + " world"')
        result = sandbox.run()

        assert result.output == "hello world"
        assert result.print_statements == []

    def test_list_operations(self):
        """Test list operations in sandbox."""
        sandbox = PythonSandbox("[1, 2, 3] + [4, 5]")
        result = sandbox.run()

        assert result.output == [1, 2, 3, 4, 5]
        assert result.print_statements == []

    def test_dict_operations(self):
        """Test dictionary operations in sandbox."""
        sandbox = PythonSandbox("{'a': 1, 'b': 2}")
        result = sandbox.run()

        assert result.output == {"a": 1, "b": 2}
        assert result.print_statements == []

    def test_function_definition_and_call(self):
        """Test function definition and calling."""
        sandbox = PythonSandbox("""
def add(a, b):
    return a + b

add(5, 3)
""")
        result = sandbox.run()

        assert result.output == 8
        assert result.print_statements == []

    def test_complex_expression(self):
        """Test complex expressions."""
        sandbox = PythonSandbox("""
x = 10
y = 20
z = x * y + 5
print(f"Result: {z}")
z
""")
        result = sandbox.run()

        assert result.output == 205
        assert result.print_statements == ["Result: 205"]

    def test_input_with_complex_types(self):
        """Test inputs with complex data types."""
        sandbox = PythonSandbox(
            """
print(f"List length: {len(data)}")
sum(data)
""",
            inputs=["data"],
        )
        result = sandbox.run(data=[1, 2, 3, 4, 5])

        assert result.output == 15
        assert result.print_statements == ["List length: 5"]

    def test_print_with_newlines(self):
        """Test print statements with newlines."""
        sandbox = PythonSandbox('print("Line 1\\nLine 2"); 42')
        result = sandbox.run()

        assert result.output == 42
        assert result.print_statements == ["Line 1\nLine 2"]
