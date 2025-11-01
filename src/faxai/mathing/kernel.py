from __future__ import annotations

from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import logging

from faxai.mathing.bandwidth import Bandwidth

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
        Build a kernel with the given bandwidth.
        Args:
            bandwidth: The bandwidth of the kernel.

        If bandwidth is None, it must be provided when applying the kernel.
        """
        self._bandwidth = bandwidth


    def set_bandwidth(self, bandwidth: Bandwidth):
        """
        Set the bandwidth of the kernel.

        Args:
            bandwidth: The bandwidth of the kernel.
        """
        self._bandwidth = bandwidth


    @abstractmethod
    def _apply(self, x: float) -> float:
        """
        Apply the kernel function to the given distance.

        Bandwidth agnostic function to simplify the implementation of stationary kernels.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement _apply method.")


    def apply(self, a: np.array, b: np.array, bandwidth: Bandwidth | None = None) -> float:
        """
        Apply the kernel function to the given points a and b with the specified bandwidth.

        Args:
            a: First point.
            b: Second point.
            bandwidth: The bandwidth of the kernel. If None, the kernel's bandwidth is used.

        Returns:
            The result of applying the kernel function. A non-negative real number.
        """
        bandwidth = bandwidth if bandwidth is not None else self._bandwidth

        # Calculate distance to apply kernel
        d = a - b
        x = d.T @ np.linalg.inv(bandwidth) @ d

        # Apply specific kernel
        k = self._apply(x)

        # Normalize by bandwidth determinant
        result = k / math.sqrt(np.linalg.det(bandwidth))

        return result


    def __str__(self) -> str:
        return f"{self.__class__.__name__}_{self.bandwidth:.3f}"


    def maximum(self, bandwidth: float=None) -> float:
        return self.apply(0, 0, bandwidth=bandwidth)



class UnivariateKernel(Kernel):
    """
    1D kernel for univariate data.

    This is a specialization of the Kernel class for one-dimensional data.
    It wraps the functions to handle single float inputs instead of arrays.
    """

    def __init__(self, bandwidth: Bandwidth | float | None = None):
        """
        Build a kernel with the given bandwidth.

        Args:
            bandwidth: The bandwidth of the kernel.

        If bandwidth is None, it must be provided when applying the kernel.
        """
        # If bandwidth is a float, convert it to UnivariateBandwidth
        if isinstance(bandwidth, float):
            bandwidth = Bandwidth.build_univariate(bandwidth)

        # Convert bandwidth to numpy matrix 1x1
        super().__init__(bandwidth=bandwidth)


    def apply(self, a: float, b: float, bandwidth: Bandwidth | float | None = None) -> float:
        """
        Apply the kernel function to the given points a and b with the specified bandwidth.

        Args:
            a: First point.
            b: Second point.
            bandwidth: The bandwidth of the kernel. If None, the kernel's bandwidth is used.

        Returns:
            The result of applying the kernel function. A non-negative real number.
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

    def _apply(self, a: float) -> float:
        return 0.5 if abs(a) <= 1 else 0


class TriangularKernel(Kernel):
    """
    Triangular kernel function.

    The triangular kernel decreases linearly from 1 at the center to 0 at the edges.
    """

    def _apply(self, x: float) -> float:
        return max(0, 1 - abs(x))


class EpanechnikovKernel(Kernel):
    """
    Epanechnikov kernel function.

    The Epanechnikov kernel decreases quadratically from 1 at the center to 0 at the edges.
    """

    def _apply(self, x: float) -> float:
        return max(0, 1 - x**2) * 3 / 4


class GaussianKernel(Kernel):
    """
    Gaussian kernel function.

    The Gaussian kernel is defined as the standard normal distribution function.
    This kernel is continuous and smooth over the entire real line.
    """

    def _apply(self, x: float) -> float:
        return np.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class DeltaKernel(UniformKernel):
    """
    Delta kernel function.

    The delta kernel approximates the Dirac delta function.
    This kernel is used to represent an infinitely sharp peak at a single point 0.

    Implementation note:
        This kernel is defined as a uniform kernel with an extremely small bandwidth.
    """

    def __init__(self, dimension):
        super().__init__(bandwidth=Bandwidth.build_delta(dimension))


class InfiniteKernel(UniformKernel):
    """
    Infinite kernel function.

    The infinite kernel represents a uniform distribution over the entire real line.

    Implementation note:
        This kernel is defined as a uniform kernel with an extremely large bandwidth.
    """

    def __init__(self, dimension):
        super().__init__(bandwidth=Bandwidth.build_infinite(dimension))


def create_default_kernel(df: pd.DataFrame) -> Kernel:
    """
    Create a default Gaussian kernel using Silverman's rule of thumb for bandwidth selection.
    """
    bandwidth = Bandwidth.reckon_silverman_bandwidth_from_data(df)
    return GaussianKernel(bandwidth)
