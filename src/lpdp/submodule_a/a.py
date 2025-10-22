"""
Module A - Feature attribution core implementation.
"""


def function_a(x: int) -> int:
    """
    Example function in module A.

    Args:
        x: Input parameter

    Returns:
        Processed result
    """
    return x * 2


class ClassA:
    """Example class in module A."""

    def __init__(self, value: int):
        """
        Initialize ClassA.

        Args:
            value: Initial value
        """
        self.value = value

    def method_a(self) -> int:
        """
        Example method.

        Returns:
            The stored value
        """
        return self.value
