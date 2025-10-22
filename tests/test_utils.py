"""
Tests for utils module.
"""

import pytest

from lpdp import utils


def test_helper_function():
    """Test helper_function basic functionality."""
    assert utils.helper_function(5, 10) == 15
    assert utils.helper_function(0, 0) == 0
    assert utils.helper_function(-5, 5) == 0
