"""
Tests for submodule_b functionality.
"""

from lpdp.submodule_b.b import ClassB, function_b


def test_function_b():
    """Test function_b basic functionality."""
    assert function_b(5) == 15
    assert function_b(0) == 10
    assert function_b(-10) == 0


def test_function_b_with_float():
    """Test function_b with float values."""
    assert function_b(2.5) == 12.5
    assert function_b(0.5) == 10.5


def test_class_b_initialization(sample_data):
    """Test ClassB initialization."""
    obj = ClassB(sample_data)
    assert obj.data == sample_data


def test_class_b_method_b():
    """Test ClassB method_b."""
    test_data = {"key": "value"}
    obj = ClassB(test_data)
    assert obj.method_b() == test_data

    test_list = [1, 2, 3]
    obj2 = ClassB(test_list)
    assert obj2.method_b() == test_list


def test_class_b_multiple_instances():
    """Test multiple ClassB instances."""
    obj1 = ClassB("data1")
    obj2 = ClassB("data2")

    assert obj1.method_b() == "data1"
    assert obj2.method_b() == "data2"
