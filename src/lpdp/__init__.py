"""
lpdp - Feature Attribution Study Python library for Machine Learning Explainability.

A Python library for Local-PDP (Local Partial Dependence Plot) analysis.
"""

__version__ = "0.1.0"
__author__ = "jparisu"

import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

from . import submodule_a, utils

__all__ = [
    "submodule_a",
    "utils",
]
