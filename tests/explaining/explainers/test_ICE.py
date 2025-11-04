"""
Tests for Individual Conditional Expectation (ICE) explainer.

This module contains tests for the ICE class using small ad-hoc datasets and models
to verify that calculations are correct.
"""

import numpy as np
import pandas as pd
import pytest

from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.DataCore import DataCore
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.data.DataHolder import HyperPlanes, Grid


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


class Test_ICE_SingleFeature:
    """Test ICE with a single feature to verify correctness."""
    
    def test_ice_single_feature_calculation(self):
        """Test ICE calculation for a single feature with known values."""
        # Create a dataset with 2 features but model only uses x1
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0],
            "x2": [0.0, 0.0, 0.0],  # Dummy feature not used by model
        })
        
        model = MockSingleFeatureModel()
        
        # Create ExplainerCore
        core = ExplainerCore(dataframe_X=df_X, model=model)
        
        # Configure for x1 with specific values (x2 is non-study feature)
        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 1.0, 2.0])},
            feature_limits={"x1": (0.0, 2.0)},
            use_default=False,
        )
        
        core.add_configuration("test_conf", conf)
        
        # Get ICE
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Verify it's a HyperPlanes object
        assert isinstance(ice, HyperPlanes)
        
        # Verify grid
        assert isinstance(ice.grid, Grid)
        assert len(ice.grid.grid) == 1
        np.testing.assert_array_equal(ice.grid.grid[0], np.array([0.0, 1.0, 2.0]))
        
        # Verify predictions shape: (n_grid_points, n_instances)
        # We have 3 grid points and 3 instances
        assert ice.targets.shape == (3, 3)
        
        # Verify actual predictions
        # ICE shape is (n_grid_points, n_instances)
        # For each grid point (row), we have predictions for all instances
        # Grid point 0 (x1=0): [5, 5, 5] for instances 0,1,2
        # Grid point 1 (x1=1): [15, 15, 15] for instances 0,1,2
        # Grid point 2 (x1=2): [25, 25, 25] for instances 0,1,2
        expected = np.array([
            [5.0, 5.0, 5.0],
            [15.0, 15.0, 15.0],
            [25.0, 25.0, 25.0],
        ])
        
        np.testing.assert_array_almost_equal(ice.targets, expected)
    
    def test_ice_shape_consistency(self):
        """Test that ICE output shape is consistent."""
        df_X = pd.DataFrame({
            "x1": [0.0, 0.5, 1.0, 1.5, 2.0],
            "x2": [1.0, 2.0, 3.0, 4.0, 5.0],  # Varying dummy feature
        })
        
        model = MockSingleFeatureModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)
        
        # Use bins parameter for default configuration
        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            bins=10,
            strict_limits=True,
        )
        
        core.add_configuration("test_conf", conf)
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Shape should be (n_grid_points, n_instances) = (bins, 5)
        assert ice.targets.shape == (10, 5)
        
        # Grid should have 1 dimension with 10 points
        assert len(ice.grid.grid) == 1
        assert ice.grid.grid[0].shape == (10,)


class Test_ICE_MultiFeature:
    """Test ICE with multiple features."""
    
    def test_ice_two_features_calculation(self):
        """Test ICE calculation with two features."""
        # Create dataset with 2 features
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0],
            "x2": [0.0, 1.0],
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
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Shape: (n_grid_points, n_instances) = (3, 2)
        assert ice.targets.shape == (3, 2)
        
        # Grid point 0 (x1=0): Instance 0 (x2=0) -> 5, Instance 1 (x2=1) -> 8
        # Grid point 1 (x1=1): Instance 0 (x2=0) -> 7, Instance 1 (x2=1) -> 10
        # Grid point 2 (x1=2): Instance 0 (x2=0) -> 9, Instance 1 (x2=1) -> 12
        expected = np.array([
            [5.0, 8.0],
            [7.0, 10.0],
            [9.0, 12.0],
        ])
        
        np.testing.assert_array_almost_equal(ice.targets, expected)
    
    def test_ice_with_three_features(self):
        """Test ICE with three features, studying one."""
        df_X = pd.DataFrame({
            "x1": [0.0, 1.0, 2.0],
            "x2": [1.0, 2.0, 3.0],
            "x3": [0.0, 0.0, 0.0],  # Not used by model but present
        })
        
        model = MockLinearModel()
        core = ExplainerCore(dataframe_X=df_X, model=model)
        
        conf = ExplainerConfiguration(
            datacore=core.datacore(),
            study_features=["x1"],
            feature_values={"x1": np.array([0.0, 2.0])},
            feature_limits={"x1": (0.0, 2.0)},
            use_default=False,
        )
        
        core.add_configuration("test_conf", conf)
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Shape: (n_grid_points, n_instances) = (2, 3)
        assert ice.targets.shape == (2, 3)
        
        # Grid point 0 (x1=0): Instance 0 (x2=1) -> 8, Instance 1 (x2=2) -> 11, Instance 2 (x2=3) -> 14
        # Grid point 1 (x1=2): Instance 0 (x2=1) -> 12, Instance 1 (x2=2) -> 15, Instance 2 (x2=3) -> 18
        expected = np.array([
            [8.0, 11.0, 14.0],
            [12.0, 15.0, 18.0],
        ])
        
        np.testing.assert_array_almost_equal(ice.targets, expected)


class Test_ICE_EdgeCases:
    """Test edge cases for ICE."""
    
    def test_ice_single_instance(self):
        """Test ICE with a single data instance."""
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
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Shape should be (n_grid_points, n_instances) = (3, 1)
        assert ice.targets.shape == (3, 1)
        
        # Predictions: [[5], [15], [25]]
        expected = np.array([[5.0], [15.0], [25.0]])
        np.testing.assert_array_almost_equal(ice.targets, expected)
    
    def test_ice_single_grid_point(self):
        """Test ICE with a single grid point."""
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
        ice = core.explain(technique="ice", configuration="test_conf")
        
        # Shape should be (n_grid_points, n_instances) = (1, 3)
        assert ice.targets.shape == (1, 3)
        
        # All instances at x1=1.5 -> prediction = 20
        expected = np.array([[20.0, 20.0, 20.0]])
        np.testing.assert_array_almost_equal(ice.targets, expected)
