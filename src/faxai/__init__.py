"""
faxai - Feature Attribution Study Python library for Machine Learning Explainability.

A Python library for feature attribution model agnostic explainability methods in machine learning.
"""

__version__ = "0.1.0"
__author__ = "jparisu"

import logging

from . import mathing, utils

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "mathing",
    "utils",
]
