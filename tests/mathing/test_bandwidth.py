"""
Exhaustive tests for the Bandwidth class and its methods.
"""

import math

import numpy as np
import pandas as pd
import pytest

from faxai.mathing.bandwidth import Bandwidth


class Test_BandwidthCheckMatrix:
    """Test check_bandwidth_matrix method."""

    Valid_Matrices = [
        np.array([[1.0, 0.1], [0.1, 1.0]]),
        np.array([[2.0]]),
        np.array([[3.0, 0.5, 0.2], [0.5, 4.0, 0.3], [0.2, 0.3, 5.0]]),
    ]

    Invalid_Matrices = [
        np.array([[1.0, 2.0], [3.0, 1.0]]),  # Non-symmetric
        np.array([[1.0, 0.0], [0.0, -1.0]]),  # Non-positive
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),  # Non-square
        np.array([[1.0, 0.0], [0.0, 0.0]]),  # Near-zero determinant
        np.array([2.0]),  # Not 2D
    ]

    def test_check_bandwidth_matrix(self):
        """Test check_bandwidth_matrix with valid and invalid matrices."""

        for matrix in self.Valid_Matrices:
            assert Bandwidth.check_bandwidth_matrix(matrix, throw=False) is True

        for matrix in self.Invalid_Matrices:
            assert Bandwidth.check_bandwidth_matrix(matrix, throw=False) is False

        # Test throw
        for matrix in self.Invalid_Matrices:
            with pytest.raises(ValueError):
                Bandwidth.check_bandwidth_matrix(matrix, throw=True)

    def test_check_bandwidth_matrix_efficient_determinant(self):
        """Test check_bandwidth_matrix with efficient determinant output in argument."""
        matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
        determinant = []
        is_valid = Bandwidth.check_bandwidth_matrix(matrix, throw=False, determinant_output=determinant)

        assert is_valid is True
        assert len(determinant) == 1
        assert math.isclose(determinant[0], 6.0)


class Test_BandwidthMethods:
    """Test Bandwidth getter methods."""

    def test_matrix_returns_correct_matrix(self):
        """Test that matrix() returns the correct matrix."""
        matrix = np.array([[2.0, 0.1], [0.1, 3.0]])
        bandwidth = Bandwidth(matrix)
        assert np.array_equal(bandwidth.matrix(), matrix)

    def test_determinant_computed_correctly(self):
        """Test that determinant is computed correctly."""
        matrix = np.array([[2.0, 0.1], [0.1, 3.0]])
        bandwidth = Bandwidth(matrix)
        expected_det = 2.0 * 3.0 - 0.1 * 0.1
        assert np.isclose(bandwidth.determinant(), expected_det)

    def test_determinant_caching(self):
        """Test that determinant is cached."""
        matrix = np.array([[2.0, 0.1], [0.1, 3.0]])
        bandwidth = Bandwidth(matrix)
        det1 = bandwidth.determinant()
        det2 = bandwidth.determinant()
        assert det1 == det2
        assert bandwidth._determinant is not None

    def test_inverse_computed_correctly(self):
        """Test that inverse is computed correctly."""
        matrix = np.array([[2.0, 0.1], [0.1, 3.0]])
        bandwidth = Bandwidth(matrix)
        inv = bandwidth.inverse()
        # Check that matrix * inverse = identity
        product = bandwidth.matrix() @ inv
        assert np.allclose(product, np.eye(2))

    def test_inverse_caching(self):
        """Test that inverse is cached."""
        matrix = np.array([[2.0, 0.1], [0.1, 3.0]])
        bandwidth = Bandwidth(matrix)
        inv1 = bandwidth.inverse()
        inv2 = bandwidth.inverse()
        assert np.array_equal(inv1, inv2)
        assert bandwidth._inverse is not None

    def test_inverse_identity_matrix(self):
        """Test inverse of near-identity matrix."""
        # Use a matrix close to identity but with positive off-diagonal
        matrix = np.array([[1.0, 0.01, 0.01], [0.01, 1.0, 0.01], [0.01, 0.01, 1.0]])
        bandwidth = Bandwidth(matrix)
        inv = bandwidth.inverse()
        product = bandwidth.matrix() @ inv
        assert np.allclose(product, np.eye(3))


class Test_BandwidthVarianceMethods:
    """Test variance-based bandwidth construction methods."""

    def test_reckon_variance_bandwidth_default_proportion(self):
        """Test variance bandwidth with default proportion."""
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        bandwidth = Bandwidth.reckon_variance_bandwidth(sigma)
        assert np.array_equal(bandwidth.matrix(), sigma)

    def test_reckon_variance_bandwidth_custom_proportion(self):
        """Test variance bandwidth with custom proportion."""
        sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
        proportion = 0.5
        bandwidth = Bandwidth.reckon_variance_bandwidth(sigma, proportion)
        assert np.array_equal(bandwidth.matrix(), sigma * proportion)

    def test_reckon_variance_bandwidth_from_data(self):
        """Test variance bandwidth from DataFrame."""
        # Create DataFrame with singular covariance
        data = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 5, 6, 7, 10],  # Non-perfect linear relation with x1
            }
        )

        bandwidth = Bandwidth.reckon_variance_bandwidth_from_data(data)
        expected_cov = data.cov().values
        assert np.array_equal(bandwidth.matrix(), expected_cov)

    def test_reckon_variance_bandwidth_from_data_with_proportion(self):
        """Test variance bandwidth from DataFrame with proportion."""
        data = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 5, 6, 7, 10],  # Non-perfect linear relation with x1
            }
        )
        proportion = 0.3
        bandwidth = Bandwidth.reckon_variance_bandwidth_from_data(data, proportion)
        expected_cov = data.cov().values * proportion
        assert np.array_equal(bandwidth.matrix(), expected_cov)


class Test_BandwidthSilvermanMethods:
    """Test Silverman bandwidth construction methods."""

    def test_reckon_silverman_bandwidth_1d(self):
        """Test Silverman bandwidth for 1D data."""
        samples = 100
        sigma = np.array([1.5])
        bandwidth = Bandwidth.reckon_silverman_bandwidth(samples, sigma)

        expected_h = samples ** (-1 / 5) * sigma[0]
        expected_matrix = np.array([[expected_h**2]])
        assert np.allclose(bandwidth.matrix(), expected_matrix)

    def test_reckon_silverman_bandwidth_2d(self):
        """Test Silverman bandwidth for 2D data."""
        samples = 100
        sigma = np.array([1.0, 2.0])
        bandwidth = Bandwidth.reckon_silverman_bandwidth(samples, sigma)

        factor0 = samples ** (-1 / 6) * sigma[0]
        factor1 = samples ** (-1 / 6) * sigma[1]
        expected_matrix = np.array([[factor0**2, 0.0], [0.0, factor1**2]])

        assert np.allclose(bandwidth.matrix(), expected_matrix)

    def test_reckon_silverman_bandwidth_from_data(self):
        """Test Silverman bandwidth from DataFrame."""
        data = pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5],
                "x2": [2, 5, 6, 7, 10],  # Non-perfect linear relation with x1
            }
        )

        bandwidth = Bandwidth.reckon_silverman_bandwidth_from_data(data)
        samples = len(data)

        sigma = data.std().values

        factor0 = samples ** (-1 / 6) * sigma[0]
        factor1 = samples ** (-1 / 6) * sigma[1]
        expected_matrix = np.array([[factor0**2, 0.0], [0.0, factor1**2]])

        assert np.allclose(bandwidth.matrix(), expected_matrix)


class Test_BandwidthSpecialMethods:
    """Test special methods (__eq__, __getitem__, __str__)."""

    def test_eq_equal_bandwidths(self):
        """Test equality of identical bandwidths."""
        matrix = np.array([[1.0, 0.1], [0.1, 1.0]])
        b1 = Bandwidth(matrix)
        b2 = Bandwidth(matrix.copy())
        assert b1 == b2

    def test_eq_different_bandwidths(self):
        """Test inequality of different bandwidths."""
        b1 = Bandwidth(np.array([[1.0, 0.1], [0.1, 1.0]]))
        b2 = Bandwidth(np.array([[2.0, 0.1], [0.1, 2.0]]))
        assert b1 != b2

    def test_eq_with_non_bandwidth_object(self):
        """Test equality with non-Bandwidth object."""
        bandwidth = Bandwidth(np.array([[1.0]]))
        result1 = bandwidth == "not a bandwidth"
        result2 = bandwidth == 42
        assert result1 == NotImplemented or not result1
        assert result2 == NotImplemented or not result2

    def test_getitem_single_element(self):
        """Test accessing single element."""
        matrix = np.array([[1.0, 0.5], [0.5, 4.0]])
        bandwidth = Bandwidth(matrix)
        assert bandwidth[0, 0] == 1.0
        assert bandwidth[0, 1] == 0.5
        assert bandwidth[1, 0] == 0.5
        assert bandwidth[1, 1] == 4.0

    def test_getitem_row(self):
        """Test accessing a row."""
        matrix = np.array([[1.0, 0.5], [0.5, 4.0]])
        bandwidth = Bandwidth(matrix)
        assert np.array_equal(bandwidth[0], np.array([1.0, 0.5]))

    def test_str_representation(self):
        """Test string representation."""
        matrix = np.array([[1.0, 0.1], [0.1, 2.0]])
        bandwidth = Bandwidth(matrix)
        str_repr = str(bandwidth)
        assert "1." in str_repr
        assert "2." in str_repr


class Test_BandwidthBuildMethods:
    """Test static build methods for special bandwidth matrices."""

    def test_build_univariate_scalar(self):
        """Test building univariate bandwidth."""
        h = 1.5
        bandwidth = Bandwidth.build_univariate(h)
        assert bandwidth.matrix().shape == (1, 1)
        assert bandwidth.matrix()[0, 0] == h

    def test_build_univariate_different_values(self):
        """Test building univariate bandwidth with different values."""
        for h in [0.1, 1.0, 5.0, 10.0]:
            bandwidth = Bandwidth.build_univariate(h)
            assert bandwidth.matrix()[0, 0] == h

    def test_build_diagonal_1d(self):
        """Test building diagonal bandwidth matrix (1D)."""
        diagonal = np.array([2.0])
        bandwidth = Bandwidth.build_diagonal(diagonal)
        assert bandwidth.matrix().shape == (1, 1)
        assert bandwidth.matrix()[0, 0] == 2.0


class Test_BandwidthEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nearly_symmetric_matrix_accepted(self):
        """Test that nearly symmetric matrix (within tolerance) is accepted."""
        # Create a matrix that's symmetric within numerical precision
        matrix = np.array([[1.0, 0.5 + 1e-15], [0.5, 2.0]])
        bandwidth = Bandwidth(matrix)
        assert bandwidth is not None

    def test_covariance_like_matrix(self):
        """Test with a realistic covariance-like matrix."""
        matrix = np.array([[4.0, 1.5], [1.5, 3.0]])
        bandwidth = Bandwidth(matrix)
        assert bandwidth.determinant() > 0
        assert np.allclose(bandwidth.matrix(), bandwidth.matrix().T)
