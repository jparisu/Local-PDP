"""
Tests for utils module.
"""

import pytest

from yourlib import utils


def test_helper_function():
    """Test helper_function basic functionality."""
    assert utils.helper_function(5, 10) == 15
    assert utils.helper_function(0, 0) == 0
    assert utils.helper_function(-5, 5) == 0


def test_helper_function_with_strings():
    """Test helper_function with string concatenation."""
    assert utils.helper_function("Hello", " World") == "Hello World"
    assert utils.helper_function("", "test") == "test"


def test_validate_input_with_valid_data():
    """Test validate_input with valid data."""
    assert utils.validate_input("data") is True
    assert utils.validate_input(42) is True
    assert utils.validate_input([1, 2, 3]) is True
    assert utils.validate_input({"key": "value"}) is True


def test_validate_input_with_none():
    """Test validate_input with None."""
    assert utils.validate_input(None) is False


def test_process_data_with_valid_data():
    """Test process_data with valid data."""
    data = {"key": "value"}
    assert utils.process_data(data) == data

    list_data = [1, 2, 3]
    assert utils.process_data(list_data) == list_data


def test_process_data_with_invalid_data():
    """Test process_data with invalid data."""
    with pytest.raises(ValueError, match="Invalid input data"):
        utils.process_data(None)
