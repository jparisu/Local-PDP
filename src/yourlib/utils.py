"""
Utility functions for yourlib.
"""


def helper_function(a, b):
    """
    Helper function for common operations.

    Args:
        a: First parameter
        b: Second parameter

    Returns:
        Combined result
    """
    return a + b


def validate_input(data):
    """
    Validate input data.

    Args:
        data: Data to validate

    Returns:
        True if valid, False otherwise
    """
    if data is None:
        return False
    return True


def process_data(data):
    """
    Process input data.

    Args:
        data: Data to process

    Returns:
        Processed data
    """
    if not validate_input(data):
        raise ValueError("Invalid input data")
    return data
