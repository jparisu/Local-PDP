"""
Submodule for probability distributions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from faex.mathing.RandomGenerator import RandomGenerator


class Distribution(ABC):
    """
    Base class for probability distributions.

    This class provides methods to compute different aspects of a probability distribution:
    - statistical properties: mean, variance, mode, median
    - probability functions: PDF, CDF
    - sampling methods: random sampling
    - etc.

    It also implements caching for performance optimization.

    Dev note:
        Subclasses should use the `@cache_method` decorator for efficient caching of computed properties.
    """

    ############################
    # Statistics

    @abstractmethod
    def mean(self) -> float:
        """
        Calculate the mean of the distribution.
        """
        pass

    @abstractmethod
    def std(self, ddof: int = 0) -> float:
        """
        Calculate the standard deviation of the distribution.
        """
        pass

    @abstractmethod
    def moded(self) -> float:
        """
        Calculate the mode of the distribution.
        """
        pass

    @abstractmethod
    def median(self) -> float:
        """
        Calculate the median of the distribution.
        """
        pass

    ############################
    # Probability Functions

    @abstractmethod
    def pdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate the probability density function (PDF).
        """
        pass

    @abstractmethod
    def maximum_pdf(self) -> float:
        """
        Calculate the maximum value of the PDF.
        """
        pass

    @abstractmethod
    def cdf(self, x: np.ndarray[float]) -> np.ndarray[float]:
        """
        Calculate the cumulative distribution function (CDF).
        """
        pass

    ############################
    # Sampling Methods

    @abstractmethod
    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray[float]:
        """
        Generate random samples from the distribution.
        """
        pass
