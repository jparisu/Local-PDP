from __future__ import annotations

import math
import numpy as np
import pandas as pd

"""Extreme values for bandwidth matrices."""
EXTREME_BANDWIDTH_VALUE = 1e16


class Bandwidth:
    """
    Base class for bandwidth matrix element.

    A bandwidth is a NxN matrix that controls the smoothness of the kernel density estimation (KDE).
    It requires the following properties:

    1. Positive definite.
    2. Symmetric.
    3. Square NxN matrix for N-dimensional data.
    4. Non null determinant.

    This class provides an interface for efficient matrix computation and different bandwidth selection methods.
    """

    def __init__(self, matrix: np.array):
        """
        Build a bandwidth with the given matrix.

        Args:
            matrix: The bandwidth matrix.
        """

        self._matrix : np.array = matrix

        # Cache values
        self._determinant : float = None
        self._inverse : np.array = None

        # Validate bandwidth matrix
        self.check_bandwidth_matrix(throw=True)


    def check_bandwidth_matrix(self, throw: bool = True):
        """
        Check if the given matrix is a valid bandwidth matrix.

        Args:
            matrix: The bandwidth matrix to check.

        Raises:
            ValueError: If the matrix is not positive definite, not symmetric, or not NxN.
        """

        # Check matrix is square
        if self._matrix.shape[0] != self._matrix.shape[1]:
            if throw:
                raise ValueError("Bandwidth must be a square matrix.")
            return False

        # Check bandwidth is positive
        if np.any(self._matrix <= 0):
            if throw:
                raise ValueError("Bandwidth must be positive.")
            return False

        # Check bandwidth is symmetric
        if not np.allclose(self._matrix, self._matrix.T):
            if throw:
                raise ValueError("Bandwidth must be symmetric.")
            return False

        # Check bandwidth has non null determinant
        if math.isclose(self.determinant(), 0.0):
            if throw:
                raise ValueError("Bandwidth must have non null determinant.")
            return False

        return True


    def matrix(self) -> np.array:
        """
        Get the bandwidth matrix.

        Returns:
            The bandwidth matrix.
        """

        return self._matrix


    def inverse(self) -> np.array:
        """
        Get the inverse of the bandwidth matrix.

        Returns:
            The inverse of the bandwidth matrix.

        Note:
            Cached for efficiency.
        """

        if self._inverse is None:
            self._inverse = np.linalg.inv(self._matrix)

        return self._inverse


    def determinant(self) -> float:
        """
        Get the determinant of the bandwidth matrix.

        Returns:
            The determinant of the bandwidth matrix.

        Note:
            Cached for efficiency.
        """

        if self._determinant is None:
            self._determinant = np.linalg.det(self._matrix)

        return self._determinant


    ######################################################
    # Variance construction bandwidth methods

    @staticmethod
    def reckon_variance_bandwidth(
            sigma: np.array,
            proportion: float = 1.0) -> Bandwidth:
        """
        Reckon the variance bandwidth for multivariate data.

        Args:
            sigma: Covariance matrix of the data.
            proportion: Proportion to use over the variance.
        """
        return Bandwidth(proportion * sigma)

    @staticmethod
    def reckon_variance_bandwidth_from_data(
            df: pd.DataFrame,
            proportion: float = 1.0) -> Bandwidth:
        """
        Reckon the variance bandwidth for multivariate data from a pandas DataFrame.

        Args:
            df: The data as a pandas DataFrame.
            proportion: Proportion to use over the variance.

        Returns:
            The variance bandwidth matrix.
        """
        sigma = df.cov().to_numpy()
        return Bandwidth.reckon_variance_bandwidth(sigma, proportion)



    ######################################################
    # Silverman construction bandwidth methods

    @staticmethod
    def reckon_silverman_bandwidth(
            samples: int,
            sigma: np.array) -> Bandwidth:
        """
        Reckon the Silverman bandwidth for multivariate data.

        Args:
            samples: The number of data points.
            sigma: The standard deviation for each dimension of the data.
        """

        dimension = sigma.shape[0]
        matrix = np.zeros((dimension, dimension))

        for i in range(dimension):
            sqrt_h = samples ** (-1 / (dimension + 4)) * sigma[i]
            matrix[i, i] = sqrt_h ** 2

        return Bandwidth(matrix)


    @staticmethod
    def reckon_silverman_bandwidth_from_data(
            df: pd.DataFrame) -> Bandwidth:
        """
        Reckon the Silverman bandwidth for multivariate data from a pandas DataFrame.

        Args:
            df: The data as a pandas DataFrame.

        Returns:
            The Silverman bandwidth matrix.
        """

        n = df.shape[0]
        sigma = df.std().to_numpy()

        return Bandwidth.reckon_silverman_bandwidth(n, sigma)


    ###########################
    # Numpy matrix methods

    def __eq__(self, other: Bandwidth) -> bool:
        return np.array_equal(self._matrix, other.matrix())

    def __getitem__(self, key):
        return self._matrix[key]

    def __str__(self) -> str:
        return str(self._matrix)


    ################################
    # Special bandwidth matrices construction

    @staticmethod
    def build_identity(dimension: int) -> Bandwidth:
        """
        Build an identity bandwidth matrix of the given dimension.

        Args:
            dimension: The dimension of the identity matrix.

        Returns:
            The identity bandwidth matrix.
        """

        matrix = np.eye(dimension)
        return Bandwidth(matrix)

    @staticmethod
    def build_univariate(h: float) -> Bandwidth:
        """
        Build a univariate bandwidth matrix with the given bandwidth.
        Args:
            h: The bandwidth.
        Returns:
            The univariate bandwidth matrix.
        """
        matrix = np.array([[h]])
        return Bandwidth(matrix)

    @staticmethod
    def build_diagonal(diagonal: np.array) -> Bandwidth:
        """
        Build a diagonal bandwidth matrix with the given diagonal elements.

        Args:
            diagonal: The diagonal elements of the matrix.

        Returns:
            The diagonal bandwidth matrix.
        """

        matrix = np.diag(diagonal)
        return Bandwidth(matrix)

    @staticmethod
    def build_delta(dimension: int) -> Bandwidth:
        """
        Build a delta bandwidth matrix of the given dimension.

        Args:
            dimension: The dimension of the delta matrix.

        Returns:
            The delta bandwidth matrix.
        """

        matrix = np.diag((dimension, dimension), EXTREME_BANDWIDTH_VALUE**-1)
        return Bandwidth(matrix)

    @staticmethod
    def build_infinite(dimension: int) -> Bandwidth:
        """
        Build an infinite bandwidth matrix of the given dimension.

        Args:
            dimension: The dimension of the infinite matrix.

        Returns:
            The infinite bandwidth matrix.
        """
        matrix = np.diag((dimension, dimension), EXTREME_BANDWIDTH_VALUE)
        return Bandwidth(matrix)
