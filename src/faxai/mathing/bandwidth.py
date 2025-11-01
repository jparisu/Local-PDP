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

    def __init__(self, matrix: np.ndarray):
        """
        Initialize a bandwidth with the given matrix.

        Args:
            matrix (np.ndarray): The bandwidth matrix. Must be a positive definite,
                symmetric, square NxN matrix with non-zero determinant.

        Raises:
            ValueError: If the matrix does not satisfy the bandwidth requirements.
        """

        self._matrix: np.ndarray = matrix

        # Cache values
        self._determinant: float | None = None
        self._inverse: np.ndarray | None = None

        # Validate bandwidth matrix
        self.check_bandwidth_matrix(throw=True)


    def check_bandwidth_matrix(self, throw: bool = True) -> bool:
        """
        Check if the bandwidth matrix satisfies all required properties.

        Validates that the matrix is:
        - Square (NxN)
        - Positive (all elements > 0)
        - Symmetric
        - Has a non-zero determinant

        Args:
            throw (bool): If True, raises ValueError on validation failure.
                If False, returns False on validation failure. Defaults to True.

        Returns:
            bool: True if the matrix is valid, False otherwise (only if throw=False).

        Raises:
            ValueError: If throw=True and the matrix is invalid.
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


    def matrix(self) -> np.ndarray:
        """
        Get the bandwidth matrix.

        Returns:
            np.ndarray: The bandwidth matrix.
        """

        return self._matrix


    def inverse(self) -> np.ndarray:
        """
        Get the inverse of the bandwidth matrix.

        The result is cached for efficiency on subsequent calls.

        Returns:
            np.ndarray: The inverse of the bandwidth matrix.

        Note:
            The inverse is computed once and cached for subsequent calls.
        """

        if self._inverse is None:
            self._inverse = np.linalg.inv(self._matrix)

        return self._inverse


    def determinant(self) -> float:
        """
        Get the determinant of the bandwidth matrix.

        The result is cached for efficiency on subsequent calls.

        Returns:
            float: The determinant of the bandwidth matrix.

        Note:
            The determinant is computed once and cached for subsequent calls.
        """

        if self._determinant is None:
            self._determinant = np.linalg.det(self._matrix)

        return self._determinant


    ######################################################
    # Variance construction bandwidth methods

    @staticmethod
    def reckon_variance_bandwidth(
            sigma: np.ndarray,
            proportion: float = 1.0) -> Bandwidth:
        """
        Compute the variance bandwidth for multivariate data.

        Args:
            sigma (np.ndarray): Covariance matrix of the data.
            proportion (float): Scaling factor to apply to the variance.
                Defaults to 1.0 (no scaling).

        Returns:
            Bandwidth: The variance-based bandwidth matrix.
        """
        return Bandwidth(proportion * sigma)

    @staticmethod
    def reckon_variance_bandwidth_from_data(
            df: pd.DataFrame,
            proportion: float = 1.0) -> Bandwidth:
        """
        Compute the variance bandwidth from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The data as a pandas DataFrame.
            proportion (float): Scaling factor to apply to the variance.
                Defaults to 1.0 (no scaling).

        Returns:
            Bandwidth: The variance-based bandwidth matrix computed from the data.
        """
        sigma = df.cov().to_numpy()
        return Bandwidth.reckon_variance_bandwidth(sigma, proportion)



    ######################################################
    # Silverman construction bandwidth methods

    @staticmethod
    def reckon_silverman_bandwidth(
            samples: int,
            sigma: np.ndarray) -> Bandwidth:
        """
        Compute the Silverman bandwidth for multivariate data.

        Silverman's rule of thumb provides an optimal bandwidth for kernel density
        estimation under certain assumptions.

        Args:
            samples (int): The number of data points in the dataset.
            sigma (np.ndarray): The standard deviation for each dimension of the data.

        Returns:
            Bandwidth: The Silverman bandwidth matrix.
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
        Compute the Silverman bandwidth from a pandas DataFrame.

        Args:
            df (pd.DataFrame): The data as a pandas DataFrame.

        Returns:
            Bandwidth: The Silverman bandwidth matrix computed from the data.
        """

        n = df.shape[0]
        sigma = df.std().to_numpy()

        return Bandwidth.reckon_silverman_bandwidth(n, sigma)


    ###########################
    # Numpy matrix methods

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another Bandwidth object.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if the bandwidth matrices are equal, False otherwise.
        """
        if not isinstance(other, Bandwidth):
            return NotImplemented
        return bool(np.array_equal(self._matrix, other.matrix()))

    def __getitem__(self, key: int | tuple[int, int]) -> float | np.ndarray:
        """
        Access elements of the bandwidth matrix using indexing.

        Args:
            key (int | tuple[int, int]): The index or indices to access.

        Returns:
            float | np.ndarray: The value(s) at the specified index/indices.
        """
        return self._matrix[key]

    def __str__(self) -> str:
        """
        Return a string representation of the bandwidth matrix.

        Returns:
            str: String representation of the bandwidth matrix.
        """
        return str(self._matrix)


    ################################
    # Special bandwidth matrices construction

    @staticmethod
    def build_identity(dimension: int) -> Bandwidth:
        """
        Build an identity bandwidth matrix of the given dimension.

        Args:
            dimension (int): The dimension of the identity matrix.

        Returns:
            Bandwidth: The identity bandwidth matrix.
        """

        matrix = np.eye(dimension)
        return Bandwidth(matrix)

    @staticmethod
    def build_univariate(h: float) -> Bandwidth:
        """
        Build a univariate bandwidth matrix with the given bandwidth value.

        Args:
            h (float): The bandwidth value for the single dimension.

        Returns:
            Bandwidth: A 1x1 bandwidth matrix.
        """
        matrix = np.array([[h]])
        return Bandwidth(matrix)

    @staticmethod
    def build_diagonal(diagonal: np.ndarray) -> Bandwidth:
        """
        Build a diagonal bandwidth matrix with the given diagonal elements.

        Args:
            diagonal (np.ndarray): The diagonal elements of the matrix.

        Returns:
            Bandwidth: The diagonal bandwidth matrix.
        """

        matrix = np.diag(diagonal)
        return Bandwidth(matrix)

    @staticmethod
    def build_delta(dimension: int) -> Bandwidth:
        """
        Build a delta bandwidth matrix representing an infinitely narrow distribution.

        This creates a diagonal matrix with extremely small values to approximate
        a Dirac delta function.

        Args:
            dimension (int): The dimension of the delta matrix.

        Returns:
            Bandwidth: The delta bandwidth matrix with infinitesimally small values.
        """

        matrix = np.diag((dimension, dimension), EXTREME_BANDWIDTH_VALUE**-1)
        return Bandwidth(matrix)

    @staticmethod
    def build_infinite(dimension: int) -> Bandwidth:
        """
        Build an infinite bandwidth matrix representing an extremely wide distribution.

        This creates a diagonal matrix with extremely large values to approximate
        a uniform distribution over the entire space.

        Args:
            dimension (int): The dimension of the infinite matrix.

        Returns:
            Bandwidth: The infinite bandwidth matrix with extremely large values.
        """
        matrix = np.diag((dimension, dimension), EXTREME_BANDWIDTH_VALUE)
        return Bandwidth(matrix)
