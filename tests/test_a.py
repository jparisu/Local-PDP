"""
Tests for submodule_a functionality.
"""

from lpdp.submodule_a.a import ClassA, function_a


def test_function_a():
    """Test function_a basic functionality."""
    assert function_a(5) == 10
    assert function_a(0) == 0
    assert function_a(-3) == -6


def test_class_a_initialization(sample_value):
    """Test ClassA initialization."""
    obj = ClassA(sample_value)
    assert obj.value == sample_value


def test_class_a_method_a():
    """Test ClassA method_a."""
    obj = ClassA(42)
    assert obj.method_a() == 42

    obj2 = ClassA("test")
    assert obj2.method_a() == "test"


def test_class_a_multiple_instances():
    """Test multiple ClassA instances."""
    obj1 = ClassA(10)
    obj2 = ClassA(20)

    assert obj1.method_a() == 10
    assert obj2.method_a() == 20
