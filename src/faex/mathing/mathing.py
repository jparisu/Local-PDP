"""
Module for mathematical operations and utilities.
"""

from __future__ import annotations

import logging
import math
import numpy as np

def reckon_silverman_bandwidth(samples: int, sigma: float):
    """
    Calculate the Silverman bandwidth for a given dataset.

    Args:
        samples (int): The number of samples in the dataset.
        sigma (float): The standard deviation of the dataset.

    Returns:
        float: The calculated Silverman bandwidth.
    """
    return 1.06 * sigma * (samples ** (-1 / 5))
