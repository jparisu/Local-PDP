"""
Submodule for probability distributions.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from faex.mathing.distribution.Distribution import Distribution
from faex.mathing.RandomGenerator import RandomGenerator
from faex.utils.decorators import cache_method


class NormalDistribution(Distribution):
    """
    Represents a normal (Gaussian) distribution parameterized by its mean and standard deviation.

    This class implements the analytical properties and methods for the normal distribution:
    - Statistical properties: mean, standard deviation, mode, and median.
    - Probability functions: probability density function (PDF) and cumulative distribution function (CDF).
    - Sampling: random samples using NumPy's random number generator.

    Parameters
    ----------
    mean : float
        The expected value (μ) of the normal distribution.
    std : float
        The standard deviation (sigma) of the normal distribution. Must be positive.

    Raises
    ------
    ValueError
        If `std` is not strictly positive.
    """

    def __init__(self, mean: float, std: float):
        if std <= 0:
            raise ValueError("Standard deviation must be positive.")
        self._mean: float = mean
        self._std: float = std

    ############################
    # Statistical Properties

    def mean(self) -> float:
        """
        Return the mean (μ) of the distribution.

        Returns
        -------
        float
            The mean of the distribution.
        """
        return self._mean

    def std(self, ddof: int = 0) -> float:
        """
        Return the standard deviation (sigma) of the distribution.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom. Included for API compatibility.
            Ignored here since sigma is a fixed parameter. Default is 0.

        Returns
        -------
        float
            The standard deviation of the distribution.

        Raises
        ------
        ValueError
            If `ddof` is not zero.
        """
        if ddof != 0:
            # In a normal distribution, std is a fixed parameter.
            # ddof is included for compatibility but does not affect the result.
            raise ValueError("ddof parameter is not applicable for fixed standard deviation.")
        return self._std

    def moded(self) -> float:
        """
        Return the mode of the normal distribution.

        Returns
        -------
        float
            The mode of the distribution (equal to the mean).
        """
        return self._mean

    def median(self) -> float:
        """
        Return the median of the normal distribution.

        Returns
        -------
        float
            The median of the distribution (equal to the mean).
        """
        return self._mean

    ############################
    # Probability Functions

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function (PDF) at a given value.

        Parameters
        ----------
        x : float
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The probability density at `x`.
        """
        return norm.pdf(x, loc=self._mean, scale=self._std)

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative distribution function (CDF) at a given value.

        Parameters
        ----------
        x : float
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability up to `x`.
        """
        return norm.cdf(x, loc=self._mean, scale=self._std)

    @cache_method
    def maximum_pdf(self) -> float:
        """
        Return the maximum value of the PDF.

        Returns
        -------
        float
            The maximum value of the PDF, which occurs at the mean.
        """
        return float(self.pdf(np.array([self._mean]))[0])

    ############################
    # Sampling Methods

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the normal distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. Default is 1.
        rng : numpy.random.Generator, optional
            A NumPy random number generator. If None, a default generator is used.

        Returns
        -------
        numpy.ndarray
            Array of random samples drawn from the distribution.
        """
        if rng is None:
            rng = RandomGenerator()

        return np.array(rng.gauss(mean=self._mean, std=self._std, n=n))


class UniformDistribution(Distribution):
    """
    Represents a continuous uniform distribution parameterized by its lower and upper bounds.

    This class implements the analytical properties and methods for the uniform distribution:
    - Statistical properties: mean, standard deviation, mode, and median.
    - Probability functions: probability density function (PDF) and cumulative distribution function (CDF).
    - Sampling: random samples using NumPy’s random number generator.

    Parameters
    ----------
    low : float
        The lower bound (a) of the uniform distribution.
    high : float
        The upper bound (b) of the uniform distribution. Must satisfy `high > low`.

    Raises
    ------
    ValueError
        If `high` is not greater than `low`.
    """

    def __init__(self, low: float, high: float):
        if not high > low:
            raise ValueError("Upper bound 'high' must be greater than lower bound 'low'.")
        self._low: float = low
        self._high: float = high

    ############################
    # Statistical Properties

    def mean(self) -> float:
        """
        Return the mean of the uniform distribution.

        Returns
        -------
        float
            The mean value, computed as (a + b) / 2.
        """
        return 0.5 * (self._low + self._high)

    def std(self, ddof: int = 0) -> float:
        """
        Return the standard deviation of the uniform distribution.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom (ignored; included for API compatibility).

        Returns
        -------
        float
            The standard deviation, computed as (b - a) / sqrt(12).
        """
        if ddof != 0:
            # In a uniform distribution, std is a fixed parameter.
            # ddof is included for compatibility but does not affect the result.
            raise ValueError("ddof parameter is not applicable for fixed standard deviation.")
        return (self._high - self._low) / math.sqrt(12)

    def moded(self) -> float:
        """
        Return the mode of the uniform distribution.

        Returns
        -------
        float
            Since all values are equally likely, returns the midpoint (a + b) / 2.
        """
        return self.mean()

    def median(self) -> float:
        """
        Return the median of the uniform distribution.

        Returns
        -------
        float
            The median, which is also the midpoint (a + b) / 2.
        """
        return self.mean()

    ############################
    # Probability Functions

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the probability density function (PDF) at a given value.

        Parameters
        ----------
        x : float
            The point at which to evaluate the PDF.

        Returns
        -------
        float
            The probability density at `x`. Zero outside [a, b].
        """
        return np.where((x < self._low) | (x > self._high), 0.0, 1.0 / (self._high - self._low))

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the cumulative distribution function (CDF) at a given value.

        Parameters
        ----------
        x : float
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The cumulative probability up to `x`.
        """
        return np.where(
            x < self._low,
            0.0,
            np.where(
                x > self._high,
                1.0,
                (x - self._low) / (self._high - self._low),
            ),
        )

    def maximum_pdf(self) -> float:
        """
        Return the maximum value of the PDF.

        Returns
        -------
        float
            The constant height of the PDF: 1 / (b - a).
        """
        return 1.0 / (self._high - self._low)

    ############################
    # Sampling Methods

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the uniform distribution.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. Default is 1.
        rng : numpy.random.Generator, optional
            A NumPy random number generator. If None, a default generator is used.

        Returns
        -------
        numpy.ndarray
            Array of random samples drawn uniformly from [a, b].
        """
        if rng is None:
            rng = RandomGenerator()
        return np.array(rng.uniform(low=self._low, high=self._high, n=n))
