"""
Code blocks used in verifying model-written code.

These functions are used in the Monty sandbox environment, so they are compatible with
the constraints of that environment (e.g., no external libraries, limited built-in
libraries).

I based these functions on the ones used in HumanEval+ and MBPP+ for compatibility.
"""

from typing import Any


def is_float(value: str | list[str] | None, require_all: bool = True) -> bool:
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
    a: int | float | list[int | float],
    b: int | float | list[int | float],
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-7,
) -> bool:
    """Determines if two values or lists are close within given tolerances.

    The default values for rel_tol and abs_tol are set to match those used in the
    HumanEval+ and MBPP+ datasets. See:
    - [HumanEval+](https://huggingface.co/datasets/evalplus/humanevalplus)
    - [MBPP+](https://huggingface.co/datasets/evalplus/mbppplus)

    Args:
        a: The first value or list to compare.
        b: The second value or list to compare.
        abs_tol: The minimum absolute tolerance - useful for comparisons
            near zero. Defaults to 1e-6.
        rel_tol: The relative tolerance - the maximum allowed difference
            relative to the larger absolute value of a or b.
            Defaults to 1e-7.

    Returns:
        bool: True if all compared elements are within tolerance, False otherwise.

    Raises:
        ValueError: If both inputs are lists but have different lengths.
    """
    # Handle sets: sort elements and convert to lists
    if isinstance(a, set):
        a = sorted(a)
    if isinstance(b, set):
        b = sorted(b)

    # Check if the inputs are sequences
    a_is_seq = isinstance(a, (list, tuple))
    b_is_seq = isinstance(b, (list, tuple))

    # Case 1: Both are sequences
    if a_is_seq and b_is_seq:
        if len(a) != len(b):
            raise ValueError("Lists must have the same length.")
        return all(is_close(ai, bi, rel_tol, abs_tol) for ai, bi in zip(a, b))

    # Case 2: One is a sequence, the other is a scalar
    if a_is_seq:
        return all(is_close(ai, b, rel_tol, abs_tol) for ai in a)

    if b_is_seq:
        return all(is_close(a, bi, rel_tol, abs_tol) for bi in b)

    # Both are scalars; compute the difference
    # Handle infinite values (-inf and inf)
    if a == b:
        return True

    a, b = float(a), float(b)

    diff = abs(a - b)

    return diff <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_correct(
    output_value: Any, expected_value: Any, abs_tol: float = 0.0, rel_tol: float = 1e-7
) -> bool:
    """
    Determine whether the output value is correct by comparing it to the expected value.

    For floats (or structures containing floats), uses is_close for comparison with tolerance.
    For other types, uses direct equality (==). Handles mixed-type lists recursively.
    """
    if abs_tol == 0:
        abs_tol = 1e-6

    # Handle sets: sort elements and convert to lists for comparison
    if isinstance(expected_value, set):
        expected_value = sorted(expected_value)
    if isinstance(output_value, set):
        output_value = sorted(output_value)

    # Check if the inputs are sequences
    expected_is_seq = isinstance(expected_value, (list, tuple))
    output_is_seq = isinstance(output_value, (list, tuple))

    # Case 1: Both are sequences -> recurse on elements
    if expected_is_seq and output_is_seq:
        if len(expected_value) != len(output_value):
            return False
        return all(
            is_correct(o, e, abs_tol, rel_tol)
            for o, e in zip(output_value, expected_value)
        )

    # Case 2: One is a sequence, the other is not -> type mismatch
    if expected_is_seq or output_is_seq:
        return False

    # Case 3: Both are scalars -> decide based on expected_value type
    if is_float(expected_value):
        return is_close(output_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol)
    else:
        return output_value == expected_value


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
    return sum(test_results) / len(test_results)
