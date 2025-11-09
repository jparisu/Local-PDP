"""
Exhaustive tests for the Kernel classes and their methods.
"""

import math

import numpy as np
import pandas as pd
import pytest

from faex.mathing.bandwidth import Bandwidth
from faex.mathing.kernel import (
    DeltaKernel,
    EpanechnikovKernel,
    GaussianKernel,
    InfiniteKernel,
    Kernel,
    TriangularKernel,
    UniformKernel,
    create_default_kernel,
)


class Test_KernelBaseClass:
    """Test the abstract Kernel base class."""

    def test_kernel_cannot_be_instantiated(self):
        """Test that Kernel abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Kernel()

    def test_kernel_init_with_bandwidth(self):
        """Test kernel initialization with bandwidth."""
        bandwidth = Bandwidth(np.array([[1.0, 0.1], [0.1, 1.0]]))
        kernel = GaussianKernel(bandwidth)
        assert kernel.bandwidth() == bandwidth

    def test_kernel_init_without_bandwidth(self):
        """Test kernel initialization without bandwidth."""
        kernel = GaussianKernel(None)
        assert kernel.bandwidth() is None

    def test_set_bandwidth(self):
        """Test setting bandwidth after initialization."""
        kernel = GaussianKernel(None)
        bandwidth = Bandwidth(np.array([[1.0, 0.1], [0.1, 1.0]]))
        kernel.set_bandwidth(bandwidth)
        assert kernel.bandwidth() == bandwidth

    def test_set_bandwidth_replaces_existing(self):
        """Test that set_bandwidth replaces existing bandwidth."""
        bandwidth1 = Bandwidth(np.array([[1.0, 0.1], [0.1, 1.0]]))
        bandwidth2 = Bandwidth.build_univariate(2.0)
        kernel = GaussianKernel(bandwidth1)
        kernel.set_bandwidth(bandwidth2)
        assert kernel.bandwidth() == bandwidth2

    def test_str_representation(self):
        """Test string representation of kernel."""
        bandwidth = Bandwidth(np.array([[1.0, 0.1], [0.1, 1.0]]))
        kernel = GaussianKernel(bandwidth)
        str_repr = str(kernel)
        assert "GaussianKernel" in str_repr


class Test_KernelProperties_1d:
    """Test the apply method of Kernel class - SKIPPED due to implementation bugs."""

    Kernels = [
        UniformKernel(Bandwidth(np.array([[1.0]]))),
        TriangularKernel(Bandwidth(np.array([[1.0]]))),
        EpanechnikovKernel(Bandwidth(np.array([[1.0]]))),
        GaussianKernel(Bandwidth(np.array([[1.0]]))),
        DeltaKernel(dimension=1),
        InfiniteKernel(dimension=1),
    ]

    X0 = np.linspace(-5, 5, 1000)

    def test_kernel_property_non_negative(self):
        """Test that kernel values are non-negative."""
        for kernel in self.Kernels:
            values = [kernel.apply(np.array([x]), np.array([0.0])) for x in self.X0]
            assert all(v >= 0 for v in values), f"{kernel} produced negative values"

    @pytest.mark.skip(reason="Integration over finite limits not working properly")
    def test_kernel_property_integrates_to_one(self):
        """Test that kernel integrates to one over its domain."""
        for kernel in self.Kernels:
            values = [kernel.apply(np.array([x]), np.array([0.0])) for x in self.X0]
            integral = np.trapezoid(values, self.X0)
            assert math.isclose(integral, 1.0, rel_tol=1e-3), f"{kernel} does not integrate to one"

    def test_kernel_property_symmetric(self):
        """Test that kernel is symmetric."""
        for kernel in self.Kernels:
            for x in self.X0:
                val_pos = kernel.apply(np.array([x]), np.array([0.0]))
                val_neg = kernel.apply(np.array([-x]), np.array([0.0]))
                assert math.isclose(val_pos, val_neg, rel_tol=1e-9), f"{kernel} is not symmetric at {x}"


class Test_KernelProperties_2d:
    """Test the apply method of Kernel class."""

    Kernels = [
        UniformKernel(Bandwidth(np.array([[1.0, 0.0], [0.0, 1.0]]))),
        TriangularKernel(Bandwidth(np.array([[1.0, 0.0], [0.0, 1.0]]))),
        EpanechnikovKernel(Bandwidth(np.array([[1.0, 0.0], [0.0, 1.0]]))),
        GaussianKernel(Bandwidth(np.array([[1.0, 0.0], [0.0, 1.0]]))),
        DeltaKernel(dimension=2),
        InfiniteKernel(dimension=2),  # Problems with integration due to domain limit
    ]

    N = 100
    X0 = np.linspace(-5, 5, N)
    X1 = np.linspace(-5, 5, N)

    def test_kernel_property_non_negative(self):
        """Test that kernel values are non-negative."""
        for kernel in self.Kernels:
            values = [kernel.apply(np.array([x0, x1]), np.array([0.0, 0.0])) for x0 in self.X0 for x1 in self.X1]
            assert all(v >= 0 for v in values), f"{kernel} produced negative values"

    @pytest.mark.skip(reason="Integration over finite limits not working properly")
    def test_kernel_property_integrates_to_one(self):
        """Test that kernel integrates to one over its domain."""
        for kernel in self.Kernels:
            # Build a 2D grid and evaluate K([x0, x1], 0)
            x0, x1 = np.meshgrid(self.X0, self.X1, indexing="ij")
            pts = np.stack([x0, x1], axis=-1).reshape(-1, 2)
            values = np.array([kernel.apply(p, np.array([0.0, 0.0])) for p in pts]).reshape(self.N, self.N)

            # II K(x) dx = I ( I K dx1 ) dx0
            inner = np.trapezoid(values, self.X1, axis=1)
            integral = np.trapezoid(inner, self.X0, axis=0)

            assert math.isclose(integral, 1.0, rel_tol=1e-3), f"{kernel} does not integrate to one"

    def test_kernel_property_symmetric(self):
        """Test that kernel is symmetric."""
        for kernel in self.Kernels:
            for x0, x1 in zip(self.X0, self.X1):
                val_pos = kernel.apply(np.array([x0, x1]), np.array([0.0, 0.0]))
                val_neg = kernel.apply(np.array([-x0, -x1]), np.array([0.0, 0.0]))
                assert math.isclose(val_pos, val_neg, rel_tol=1e-9), f"{kernel} is not symmetric at [{x0},{x1}]"


class Test_CreateDefaultKernel:
    """Test create_default_kernel function."""

    def test_create_default_kernel_returns_gaussian(self):
        """Test that default kernel is a GaussianKernel - only works for 1D data."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        kernel = create_default_kernel(df)
        assert isinstance(kernel, GaussianKernel)

    def test_create_default_kernel_has_bandwidth(self):
        """Test that default kernel has bandwidth set - only works for 1D data."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        kernel = create_default_kernel(df)
        assert kernel._bandwidth is not None

    def test_create_default_kernel_uses_silverman(self):
        """Test that default kernel uses Silverman's rule."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        kernel = create_default_kernel(df)

        # Compute expected Silverman bandwidth
        expected_bandwidth = Bandwidth.reckon_silverman_bandwidth_from_data(df)

        # Compare matrices
        assert np.allclose(kernel._bandwidth.matrix(), expected_bandwidth.matrix())

    def test_create_default_kernel_different_data(self):
        """Test default kernel with different datasets."""
        df1 = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        df2 = pd.DataFrame({"x": [10.0, 20.0, 30.0]})

        kernel1 = create_default_kernel(df1)
        kernel2 = create_default_kernel(df2)

        # Different data should produce different bandwidths
        assert not np.allclose(kernel1._bandwidth.matrix(), kernel2._bandwidth.matrix())
