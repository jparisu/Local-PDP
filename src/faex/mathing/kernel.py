from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from faex.mathing.bandwidth import Bandwidth

logger = logging.getLogger(__name__)


class Kernel(ABC):
    """
    Abstract base class for kernel functions.

    A kernel function is a mathematical function with the following properties:
    1. It is non-negative: K(x) >= 0 for all x.
    2. It integrates to one over the entire real line.
    3. It is symmetric: K(x) = K(-x) for all x.

    The kernels implemented here are multi variants with size N.
    The arguments are 2 vectors of size N, and the bandwidth is a matrix of size NxN.

    Mathematical note:
        These kernels are all stationary, meaning they depend only on the distance between points,
        not on their absolute positions.
        This is a requirement for KDE.

    Implementation note:
        These kernels uses Bandwidth class for efficient matrix computation.
    """

    def __init__(self, bandwidth: Bandwidth | None = None):
        """
        Initialize a kernel with the given bandwidth.

        Args:
            bandwidth (Bandwidth | None): The bandwidth matrix of the kernel.
                If None, it must be provided when applying the kernel.
        """
        self._bandwidth: Bandwidth | None = bandwidth

    def set_bandwidth(self, bandwidth: Bandwidth) -> None:
        """
        Set the bandwidth of the kernel.

        Args:
            bandwidth (Bandwidth): The bandwidth matrix of the kernel.
        """
        self._bandwidth = bandwidth

    def bandwidth(self) -> Bandwidth | None:
        """
        Get the bandwidth of the kernel.

        Returns:
            Bandwidth | None: The bandwidth matrix of the kernel.
        """
        return self._bandwidth

    def dimension(self) -> int | None:
        """
        Get the dimension of the kernel based on its bandwidth.

        Returns:
            int | None: The dimension of the kernel, or None if bandwidth is not set.
        """
        if self._bandwidth is None:
            return None
        return self._bandwidth.dimension()

    @abstractmethod
    def _apply(self, x: float) -> float:
        """
        Apply the kernel function to the given distance.

        This is a bandwidth-agnostic function to simplify the implementation
        of stationary kernels.

        Args:
            x (float): The Mahalanobis distance between two points.

        Returns:
            float: The kernel value at the given distance.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _apply method.")

    def apply(self, a: np.ndarray, b: np.ndarray, bandwidth: Bandwidth | None = None) -> float:
        """
        Apply the kernel function to the given points with the specified bandwidth.

        Args:
            a (np.ndarray): First point as a numpy array.
            b (np.ndarray): Second point as a numpy array.
            bandwidth (Bandwidth | None): The bandwidth matrix of the kernel.
                If None, the kernel's internal bandwidth is used.

        Returns:
            float: The kernel value between the two points. A non-negative real number.
        """
        if bandwidth is None:
            bandwidth = self._bandwidth
        if bandwidth is None:
            raise ValueError("Bandwidth must be provided either in the kernel or as an argument.")

        # Calculate distance to apply kernel
        d = a - b
        x = d.T @ bandwidth.inverse() @ d

        # Apply specific kernel
        k = self._apply(x)

        # Normalize by bandwidth determinant
        result = k / math.sqrt(bandwidth.determinant())

        return result

    def __str__(self) -> str:
        """
        Return a string representation of the kernel.

        Returns:
            str: A string representation including the kernel class name and bandwidth.
        """
        return f"{self.__class__.__name__}_{self._bandwidth}"

    def maximum(self, bandwidth: Bandwidth | None = None) -> float:
        """
        Calculate the maximum value of the kernel function.

        The maximum occurs at zero distance (when a == b).

        Args:
            bandwidth (Bandwidth | None): The bandwidth matrix to use for calculation.
                If None, the kernel's internal bandwidth is used.

        Returns:
            float: The maximum kernel value.
        """
        return self.apply(np.zeros(self.dimension()), np.zeros(self.dimension()), bandwidth=bandwidth)


class UnivariateKernel(Kernel):
    """
    1D kernel for univariate data.

    This is a specialization of the Kernel class for one-dimensional data.
    It wraps the functions to handle single float inputs instead of arrays.
    """

    def __init__(self, bandwidth: Bandwidth | float | None = None):
        """
        Initialize a univariate kernel with the given bandwidth.

        Args:
            bandwidth (Bandwidth | float | None): The bandwidth of the kernel.
                Can be a Bandwidth object, a float value, or None.
                If a float is provided, it will be converted to a Bandwidth object.
                If None, it must be provided when applying the kernel.
        """
        # If bandwidth is a float, convert it to UnivariateBandwidth
        if isinstance(bandwidth, float):
            bandwidth = Bandwidth.build_univariate(bandwidth)

        # Convert bandwidth to numpy matrix 1x1
        super().__init__(bandwidth=bandwidth)

    def univariate_apply(self, a: float, b: float, bandwidth: Bandwidth | float | None = None) -> float:
        """
        Apply the kernel function to the given scalar points with the specified bandwidth.

        Args:
            a (float): First point as a scalar value.
            b (float): Second point as a scalar value.
            bandwidth (Bandwidth | float | None): The bandwidth of the kernel.
                Can be a Bandwidth object, a float value, or None.
                If None, the kernel's internal bandwidth is used.

        Returns:
            float: The kernel value between the two points. A non-negative real number.
        """
        # Convert inputs to numpy arrays
        a_array = np.array([a])
        b_array = np.array([b])

        # If bandwidth is a float, convert it to UnivariateBandwidth
        if isinstance(bandwidth, float):
            bandwidth = Bandwidth.build_univariate(bandwidth)

        return super().apply(a_array, b_array, bandwidth=bandwidth)


class UniformKernel(Kernel):
    """
    Uniform kernel function.

    The uniform kernel is defined as 0.5 if |x| <= 1, and 0 otherwise.
    """

    def _apply(self, x: float) -> float:
        """
        Apply the uniform kernel function.

        Args:
            x (float): The Mahalanobis distance between two points.

        Returns:
            float: 0.5 if the distance is less than or equal to 1, otherwise 0.
        """
        return 0.5 if abs(x) <= 1 else 0


class TriangularKernel(Kernel):
    """
    Triangular kernel function.

    The triangular kernel decreases linearly from 1 at the center to 0 at the edges.
    """

    def _apply(self, x: float) -> float:
        """
        Apply the triangular kernel function.

        Args:
            x (float): The Mahalanobis distance between two points.

        Returns:
            float: The kernel value, decreasing linearly from 1 to 0 as distance increases.
        """
        return max(0, 1 - abs(x))


class EpanechnikovKernel(Kernel):
    """
    Epanechnikov kernel function.

    The Epanechnikov kernel decreases quadratically from 1 at the center to 0 at the edges.
    """

    def _apply(self, x: float) -> float:
        """
        Apply the Epanechnikov kernel function.

        Args:
            x (float): The Mahalanobis distance between two points.

        Returns:
            float: The kernel value, decreasing quadratically from 0.75 to 0.
        """
        return max(0, 1 - x**2) * 3 / 4


class GaussianKernel(Kernel):
    """
    Gaussian kernel function.

    The Gaussian kernel is defined as the standard normal distribution function.
    This kernel is continuous and smooth over the entire real line.
    """

    def _apply(self, x: float) -> float:
        """
        Apply the Gaussian kernel function.

        Args:
            x (float): The Mahalanobis distance between two points.

        Returns:
            float: The kernel value following the standard normal distribution.
        """
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class DeltaKernel(UniformKernel):
    """
    Delta kernel function.

    The delta kernel approximates the Dirac delta function.
    This kernel is used to represent an infinitely sharp peak at a single point 0.

    Implementation note:
        This kernel is defined as a uniform kernel with an extremely small bandwidth.
    """

    def __init__(self, dimension: int):
        """
        Initialize a delta kernel with the specified dimension.

        Args:
            dimension (int): The dimensionality of the kernel.
        """
        super().__init__(bandwidth=Bandwidth.build_delta(dimension))


class InfiniteKernel(UniformKernel):
    """
    Infinite kernel function.

    The infinite kernel represents a uniform distribution over the entire real line.

    Implementation note:
        This kernel is defined as a uniform kernel with an extremely large bandwidth.
    """

    def __init__(self, dimension: int):
        """
        Initialize an infinite kernel with the specified dimension.

        Args:
            dimension (int): The dimensionality of the kernel.
        """
        super().__init__(bandwidth=Bandwidth.build_infinite(dimension))


def create_default_kernel(df: pd.DataFrame) -> Kernel:
    """
    Create a default Gaussian kernel using Silverman's rule of thumb for bandwidth selection.

    Args:
        df (pd.DataFrame): The data used to compute the optimal bandwidth.

    Returns:
        Kernel: A GaussianKernel instance with bandwidth computed using Silverman's rule.
    """
    bandwidth = Bandwidth.reckon_silverman_bandwidth_from_data(df)
    return GaussianKernel(bandwidth)
