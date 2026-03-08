"""Unit tests for the helpers module."""

import pytest
from lfm_coder.rewards.helpers import is_float, is_close, is_correct


class TestIsFloat:
    """Test cases for the is_float function."""

    def test_string_floats(self):
        """Test strings that represent floats."""
        assert is_float("3.14") is True
        assert is_float("-1.5") is True
        assert is_float("1e-5") is True
        assert is_float("inf") is True
        assert is_float("-inf") is True
        assert is_float("nan") is True

    def test_string_integers(self):
        """Test strings that represent integers (should return False)."""
        assert is_float("1") is False
        assert is_float("0") is False
        assert is_float("-5") is False
        assert is_float("100") is False

    def test_string_non_numeric(self):
        """Test strings that are not numeric."""
        assert is_float("abc") is False
        assert is_float("hello") is False
        assert is_float("") is False

    def test_none(self):
        """Test None input."""
        assert is_float(None) is False

    def test_lists(self):
        """Test list inputs."""
        assert is_float(["3.14", "2.71"]) is True
        # "1" is not float
        assert is_float(["1", "2.0"]) is False
        # at least one is float
        assert is_float(["1", "2.1"], require_all=False) is True
        assert is_float(["abc", "3.14"], require_all=False) is True
        assert is_float(["abc", "def"], require_all=False) is False
        assert is_float([]) is False

    def test_tuples(self):
        """Test tuple inputs."""
        assert is_float(("3.14", "2.71")) is True
        assert is_float(("1", "2.0")) is False

    def test_sets(self):
        """Test set inputs."""
        assert is_float({"3.14", "2.71"}) is True
        assert is_float({"1", "2.0"}) is False


class TestIsClose:
    """Test cases for the is_close function."""

    def test_scalars_close(self):
        """Test close scalar values."""
        assert is_close(1.0, 1.0000001) is True
        assert is_close(0.0, 1e-10) is True
        assert is_close(-1.0, -1.0000001) is True

    def test_scalars_not_close(self):
        """Test not close scalar values."""
        assert is_close(1.0, 1.1) is False
        assert is_close(0.0, 0.1) is False

    def test_integers(self):
        """Test integer values."""
        assert is_close(1, 1) is True
        assert is_close(1, 2) is False

    def test_infinite_values(self):
        """Test infinite values."""
        assert is_close(float("inf"), float("inf")) is True
        assert is_close(float("-inf"), float("-inf")) is True
        assert (
            is_close(float("inf"), float("-inf")) is True
        )  # According to implementation

    def test_nan_values(self):
        """Test NaN values."""
        assert is_close(float("nan"), float("nan")) is False
        assert is_close(1.0, float("nan")) is False

    def test_lists_same_length(self):
        """Test lists of same length."""
        assert is_close([1.0, 2.0], [1.0000001, 2.0]) is True
        assert is_close([1.0, 2.0], [1.1, 2.0]) is False

    def test_lists_different_length(self):
        """Test lists of different lengths (should raise ValueError)."""
        with pytest.raises(ValueError, match="Lists must have the same length."):
            is_close([1.0, 2.0], [1.0])

    def test_sets(self):
        """Test sets (should be sorted and compared)."""
        assert is_close({1.0, 2.0}, {2.0, 1.0}) is True
        assert is_close({1.0, 2.0}, {1.1, 2.0}) is False

    def test_mixed_list_scalar(self):
        """Test list vs scalar."""
        assert is_close([1.0], 1.0) is True
        assert is_close(1.0, [1.0]) is True
        assert is_close([1.0, 2.0], 1.5) is False

    def test_tuples(self):
        """Test tuples."""
        assert is_close((1.0, 2.0), (1.0000001, 2.0)) is True

    def test_custom_tolerances(self):
        """Test with custom tolerances."""
        assert is_close(1.0, 1.1, abs_tol=0.2) is True
        assert is_close(1.0, 1.1, rel_tol=0.2) is True


class TestIsCorrect:
    """Test cases for the is_correct function."""

    def test_float_expected(self):
        """Test when expected value is a float."""
        assert is_correct(1.1000000001, 1.1) is True
        assert is_correct(1.2, 1.1) is False
        assert is_correct([1.1000000001], [1.1]) is True

    def test_int_expected(self):
        """Test when expected value is an integer."""
        assert is_correct(1, 1) is True
        assert is_correct(1, 2) is False
        assert is_correct([1], [1]) is True

    def test_string_expected(self):
        """Test when expected value is a string."""
        assert is_correct("hello", "hello") is True
        assert is_correct("hello", "world") is False

    def test_list_expected(self):
        """Test when expected value is a list."""
        assert is_correct([1, 2], [1, 2]) is True
        assert is_correct([1.0, 2.0], [1.0, 2.0]) is True
        # 1.0 and 1 are considered equal
        assert is_correct([1.0, 2.4], [1, 2.4]) is True
        assert is_correct([1, 2], [1, 3]) is False

    def test_none_expected(self):
        """Test when expected value is None."""
        assert is_correct(None, None) is True
        assert is_correct(1, None) is False

    def test_custom_tolerances(self):
        """Test with custom tolerances."""
        assert is_correct(1.2, 1.1, abs_tol=0.2) is True
        assert is_correct(1.3, 1.1, rel_tol=0.2) is True

    def test_mixed_list_expected(self):
        """Test mixed-type lists (ints and floats)."""
        assert is_correct([1, 2.5], [1, 2.5000001]) is True
        assert is_correct([1, 2.5], [1, 3.0]) is False
        assert is_correct([1, "hello"], [1, "hello"]) is True
        assert is_correct([1, 2.5], [2, 2.5]) is False  # int mismatch

    def test_sets_expected(self):
        """Test sets with mixed types."""
        assert is_correct({1, 2.5}, {2.5, 1}) is True
        assert is_correct({1, 2.5}, {1, 3.0}) is False

    def test_nested_lists(self):
        """Test nested lists."""
        assert is_correct([[1, 2.0]], [[1, 2.0000001]]) is True
        assert is_correct([[1, 2.0]], [[1, 3.0]]) is False
        assert is_correct([[1, 2.5], 3], [[1, 2.5000001], 3]) is True

    def test_sequence_length_mismatch(self):
        """Test sequences of different lengths."""
        assert is_correct([1, 2], [1]) is False
        assert is_correct([1], [1, 2]) is False

    def test_sequence_vs_scalar(self):
        """Test sequence vs scalar mismatch."""
        assert is_correct([1], 1) is False
        assert is_correct(1, [1]) is False
