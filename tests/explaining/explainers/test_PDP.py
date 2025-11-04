"""
Tests for Partial Dependence Plot (PDP) explainer.

This module contains tests for the PDP class using small ad-hoc datasets and models
to verify that calculations are correct.
"""

import numpy as np
import pandas as pd

from faxai.data.DataHolder import Grid, HyperPlane
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.ExplainerCore import ExplainerCore


class MockLinearModel:
    """
    A simple mock model for testing.

    Predicts: y = 2 * x1 + 3 * x2 + 5
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return 2 * df["x1"].values + 3 * df["x2"].values + 5


class MockSingleFeatureModel:
    """
    A simple mock model with a single feature.

    Predicts: y = x1 * 10 + 5
    Ignores other features if present.
    """

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["x1"].values * 10 + 5


class Test_PDP_SingleFeature:
    """Test PDP with a single feature to verify correctness."""

    def test_pdp_single_feature_calculation(self):
        """Test PDP calculation for a single feature with known values."""
        # Create a dataset with 2 features but model only uses x1
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0],  # Dummy feature
        })

        model = MockSingleFeatureModel()

        # Create ExplainerCore
        core = ExplainerCore(dataframe_X=df_X, model=model)

        # Configure for x1 with specific values
        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0, 2.0])},
            feature_limits={"x1": (0.0, 2.0)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)

        # Get PDP
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # Verify it's a HyperPlane object
        assert isinstance(pdp, HyperPlane)

        # Verify grid
        assert isinstance(pdp.grid, Grid)
        assert len(pdp.grid.grid) == 1
        np.testing.assert_array_equal(pdp.grid.grid[0], np.array([0.0, 1.0, 2.0]))

        # Verify predictions shape: (n_grid_points,)
        assert pdp.target.shape == (3,)

        # PDP should average ICE across all instances
        # ICE for all instances at each grid point: [[5,5,5], [15,15,15], [25,25,25]]
        # Average across instances (axis=1): [5, 15, 25]
        expected = np.array([5.0, 15.0, 25.0])

        np.testing.assert_array_almost_equal(pdp.target, expected)

    def test_pdp_averages_ice(self):
        """Test that PDP correctly averages ICE values."""
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0, 3.0],
            "x2": [0.0, 0.0, 0.0, 0.0],  # Dummy feature
        })

        model = MockSingleFeatureModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)

        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0])},
            feature_limits={"x1": (0.0, 1.0)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)

        # Get both ICE and PDP
        ice = core.explain(technique="ice", configuration="test_conf")
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # PDP should be the mean of ICE across instances (axis=1)
        expected_pdp = ice.targets.mean(axis=1)

        np.testing.assert_array_almost_equal(pdp.target, expected_pdp)


class Test_PDP_MultiFeature:
    """Test PDP with multiple features."""

    def test_pdp_two_features_calculation(self):
        """Test PDP calculation with two features, studying one."""
        # Create dataset with 2 features
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0],
            "x2": [0.0, 1.0, 2.0],
        })

        model = MockLinearModel()

        core = ExplainerCore(dataframe_X=df_X, model=model)

        # Study only x1, x2 remains constant per instance
        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0, 2.0])},
            feature_limits={"x1": (0.0, 2.0)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # Shape: (n_grid_points,) = (3,)
        assert pdp.target.shape == (3,)

        # Manual calculation:
        # ICE shape is (3 grid points, 3 instances)
        # Grid point 0 (x1=0): Instance 0 (x2=0) -> 5, Instance 1 (x2=1) -> 8, Instance 2 (x2=2) -> 11
        # Grid point 1 (x1=1): Instance 0 (x2=0) -> 7, Instance 1 (x2=1) -> 10, Instance 2 (x2=2) -> 13
        # Grid point 2 (x1=2): Instance 0 (x2=0) -> 9, Instance 1 (x2=1) -> 12, Instance 2 (x2=2) -> 15
        # PDP (mean across instances, axis=1): [(5+8+11)/3, (7+10+13)/3, (9+12+15)/3] = [8, 10, 12]
        expected = np.array([8.0, 10.0, 12.0])

        np.testing.assert_array_almost_equal(pdp.target, expected)

    def test_pdp_with_varying_x2(self):
        """Test PDP where non-study feature varies significantly."""
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0],
            "x2": [0.0, 10.0],  # Large variation in x2
        })

        model = MockLinearModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)

        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0])},
            feature_limits={"x1": (0.0, 1.0)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # ICE shape is (2 grid points, 2 instances)
        # Grid point 0 (x1=0): Instance 0 (x2=0) -> 5, Instance 1 (x2=10) -> 35
        # Grid point 1 (x1=1): Instance 0 (x2=0) -> 7, Instance 1 (x2=10) -> 37
        # PDP (mean across instances, axis=1): [(5+35)/2, (7+37)/2] = [20, 22]
        expected = np.array([20.0, 22.0])

        np.testing.assert_array_almost_equal(pdp.target, expected)


class Test_PDP_EdgeCases:
    """Test edge cases for PDP."""

    def test_pdp_single_instance(self):
        """Test PDP with a single data instance."""
        df_X = pd.DataFrame({
            "x1": [1.0],
            "x2": [0.0],  # Dummy feature
        })

        model = MockSingleFeatureModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)

        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0, 2.0])},
            feature_limits={"x1": (0.0, 2.0)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # With single instance, PDP equals ICE
        assert pdp.target.shape == (3,)
        expected = np.array([5.0, 15.0, 25.0])
        np.testing.assert_array_almost_equal(pdp.target, expected)

    def test_pdp_single_grid_point(self):
        """Test PDP with a single grid point."""
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0],  # Dummy feature
        })

        model = MockSingleFeatureModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)

        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([1.5])},
            feature_limits={"x1": (1.5, 1.5)},
            use_default=False,
        )

        core.add_configuration("test_conf", conf)
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # Single grid point
        assert pdp.target.shape == (1,)

        # All instances at x1=1.5 -> prediction = 20, mean = 20
        expected = np.array([20.0])
        np.testing.assert_array_almost_equal(pdp.target, expected)

    def test_pdp_consistency_with_ice(self):
        """Test that PDP is consistent with ICE for various configurations."""
        df_X = pd.DataFrame({
            "x1": [0.0, 0.5, 1.0, 1.5, 2.0],
            "x2": [2.0, 1.5, 1.0, 0.5, 0.0],
        })

        model = MockLinearModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)

        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            bins=5,
        )

        core.add_configuration("test_conf", conf)

        ice = core.explain(technique="ice", configuration="test_conf")
        pdp = core.explain(technique="pdp", configuration="test_conf")

        # PDP should be the average of ICE across instances (axis=1)
        expected_pdp = ice.targets.mean(axis=1)
        np.testing.assert_array_almost_equal(pdp.target, expected_pdp)
