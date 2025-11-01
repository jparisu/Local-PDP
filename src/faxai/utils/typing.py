"""
Utility functions for faxai.
"""

import typing


def is_hashable(obj: object) -> bool:
    """
    Check if an object is hashable: it has a __hash__ method and can be hashed.

    Note:
        Every class is hashable by default unless it overrides __hash__ to None.
    """

    # Check if it is an instance of Hashable
    if not isinstance(obj, typing.Hashable):
        return False

    # If it is a container, check if all elements are hashable
    if isinstance(obj, (tuple, set, frozenset)):
        return all(is_hashable(item) for item in obj)

    return True
