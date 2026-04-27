"""
Code blocks used in verifying model-written code.

These functions are used in the Monty sandbox environment, so they are compatible with
the constraints of that environment (e.g., no external libraries, limited built-in
libraries).

I based these functions on the ones used in HumanEval+ and MBPP+ for compatibility.
"""

import re
from typing import Any, Literal, TypeGuard, cast


def extract_code(
    completion: str,
    strategy: Literal["last", "all"] = "last",
) -> tuple[str | None, bool]:
    """Extract code from the last fenced code block in the LLM's response.

    If no Python fenced code blocks are found, this will attempt to extract an
    unlabeled fenced code block, and finally will extract the last "```"-
    delimited block (if the code block wasn't properly closed).

    Args:
        completion: The LLM's complete response.
        strategy: The strategy to use for extracting code.
            - "last" (default): extract the last code block in the response, prefering
                the last code block with a function or class definition.
            - "all": extract all code blocks in the response.

    Returns:
        Tuple of (extracted_code, correct_format)
    """
    for pattern, is_correct_format in zip(
        [
            # Prefer complete Python code blocks
            r"```python\s*(.*?)\s*```",
            # Fall back to generic code blocks
            r"```\s*(.*?)\s*```",
            # Last resort: extract the last "```" delimited block for unclosed code blocks
            r"```(?:\w*)?\s*(.*)$",
        ],
        [
            # Only complete Python code blocks are considered correct format
            True,
            False,
            False,
        ],
    ):
        code_blocks = re.findall(pattern, completion, re.DOTALL)

        if code_blocks:
            break

    if strategy == "all":
        return "\n\n".join(code_blocks), is_correct_format

    # Assume the solution is in the last code block with a function or class
    # definition, which skips reasoning traces that precede the final solution
    # and usage examples that follow it.
    for code_block in code_blocks[::-1]:
        if "def " in code_block or "class " in code_block:
            return code_block.strip(), is_correct_format

    # Return the final code block if none had a function or class definition.
    if code_blocks:
        return code_blocks[-1].strip(), is_correct_format

    return None, is_correct_format


def is_float(
    value: str | list[str] | tuple[str, ...] | set[str] | None, require_all: bool = True
) -> bool:
    """
    Determine whether an input (list or str) is a float data type.

    If an input is a float, then we need to use is_close instead of == for comparison to
    account for numerical imprecision.
    """
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        if len(value) == 0:
            return False
        return (
            all(is_float(v, require_all=require_all) for v in value)
            if require_all
            else any(is_float(v, require_all=require_all) for v in value)
        )
    try:
        v = float(value)
        if v.is_integer():
            return False
        return True
    except ValueError:
        return False


def is_close(
    result: int
    | float
    | list[int | float]
    | tuple[int | float, ...]
    | set[int | float],
    expected: int
    | float
    | list[int | float]
    | tuple[int | float, ...]
    | set[int | float],
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-7,
) -> bool:
    """Determines if two values or lists are close within given tolerances.

    The default values for rel_tol and abs_tol are set to match those used in the
    HumanEval+ and MBPP+ datasets. See:
    - [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus)
    - [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus)

    Args:
        result: The result from the code.
        expected: The expected value (i.e., "ground truth") to compare to the result.
        abs_tol: The minimum absolute tolerance - useful for comparisons
            near zero. Defaults to 1e-6.
        rel_tol: The relative tolerance - the maximum allowed difference
            relative to the larger absolute value of `result` or `expected`.
            Defaults to 1e-7.

    Returns:
        bool: True if all compared elements are within tolerance, False otherwise.

    Raises:
        ValueError: If both inputs are lists but have different lengths.
    """
    # Handle sets: sort elements and convert to lists
    if isinstance(result, set):
        result = sorted(result)
    if isinstance(expected, set):
        expected = sorted(expected)

    # Check if the inputs are sequences
    def is_sequence(value: Any) -> TypeGuard[list[int | float] | tuple[int | float]]:
        return isinstance(value, (list, tuple))

    # Case 1: Both are sequences
    if is_sequence(result) and is_sequence(expected):
        if len(result) != len(expected):
            raise ValueError("Lists must have the same length.")
        return all(
            is_close(ai, bi, rel_tol, abs_tol) for ai, bi in zip(result, expected)
        )

    # Case 2: One is a sequence, the other is a scalar
    if is_sequence(result):
        return all(is_close(ai, expected, rel_tol, abs_tol) for ai in result)

    if is_sequence(expected):
        return all(is_close(result, bi, rel_tol, abs_tol) for bi in expected)

    # Both are scalars; compute the difference
    # Handle infinite values (-inf and inf)
    if result == expected:
        return True

    result, expected = cast(float, result), cast(float, expected)

    diff = abs(result - expected)

    return diff <= max(rel_tol * max(abs(result), abs(expected)), abs_tol)


def is_correct(
    result_value: Any, expected_value: Any, abs_tol: float = 0.0, rel_tol: float = 1e-7
) -> bool:
    """
    Determine whether the result value is correct by comparing it to the expected value.

    For floats (or structures containing floats), uses is_close for comparison with tolerance.
    For other types, uses direct equality (==). Handles mixed-type lists recursively.
    """
    if abs_tol == 0:
        abs_tol = 1e-6

    # Handle sets: sort elements and convert to lists for comparison
    if isinstance(expected_value, set):
        expected_value = sorted(expected_value)
    if isinstance(result_value, set):
        result_value = sorted(result_value)

    # Check if the inputs are sequences
    expected_is_seq = isinstance(expected_value, (list, tuple))
    result_is_seq = isinstance(result_value, (list, tuple))

    # Case 1: Both are sequences -> recurse on elements
    if expected_is_seq and result_is_seq:
        if len(expected_value) != len(result_value):
            return False
        return all(
            is_correct(out, exp, abs_tol, rel_tol)
            for out, exp in zip(result_value, expected_value)
        )

    # Case 2: One is a sequence, the other is not -> type mismatch
    if expected_is_seq or result_is_seq:
        return False

    # Case 3: Both are scalars -> decide based on expected_value type
    if is_float(expected_value):
        return is_close(result_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol)
    else:
        return result_value == expected_value


def pass_rate(test_results: list[bool]) -> float:
    """
    Calculate the pass rate given a list of boolean test results.

    Args:
        test_results: A list of boolean values where True indicates a passed test and
            False indicates a failed test.

    Returns:
        A float representing the pass rate, calculated as the number of passed tests
        divided by the total number of tests.
    """
    if not test_results:
        return 0.0
    return sum(1 for test_result in test_results if test_result) / len(test_results)
