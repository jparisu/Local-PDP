"""
faex - Feature Attribution Study Python library for Machine Learning Explainability.

A Python library for feature attribution model agnostic explainability methods in machine learning.
"""

__version__ = "0.1.0"
__author__ = "jparisu"

# This is required for registering all explanation techniques
from faex.explaining import explainers  # noqa: F401
