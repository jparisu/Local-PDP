"""
A explanation configuration class allows to set various parameters for generating explanations for
different techniques.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

from faxai.utils.decorators import cache_method
from faxai.mathing.kernel import Kernel, create_default_kernel
from faxai.explaining.DataCore import DataCore
from faxai.data.DataHolder import Grid

logger = logging.getLogger(__name__)


@dataclass
class ExplainerConfiguration:
    """
    Configuration class for explanation generation.

    This class holds various parameters that can be adjusted to customize different explanation techniques.
    This facilitates flexibility and adaptability in generating explanations for machine learning models.

    Attributes:
        lim
        num_samples (int): Number of samples to use for generating explanations.
        random_state (int, optional): Random state for reproducibility.
        additional_params (dict): Additional parameters specific to the explanation technique.
    """

    datacore: DataCore = None
    study_features: list[str] = None
    feature_limits: dict[str, tuple[float, float]] = None
    feature_values: dict[str, np.ndarray] = None
    locality_ranges: dict[np.ndarray] = None
    kernel: Kernel = None

    def __init__(
            self,
            # Required configuration
            datacore: DataCore,
            study_features: list[str],
            *,
            # Optional configuration
            feature_limits: dict[str, tuple[float, float]] | None = None,
            feature_values: dict[str, np.ndarray] | None = None,
            locality_ranges: list[float] | None = None,
            kernel: Kernel = None,
            # Default configuration arguments
            use_default: bool = True,
            bins: int = 50,
            strict_limits: bool = True,
            locality_size: dict[float] = None
    ):
        """
        Initialize the ExplanationConfiguration with the core and study features.

        Args:
            datacore (DataCore): The ExplainerCore instance containing the dataset and model.
            study_features (list[str]): List of feature names to study.
        """

        # Input stored
        self.datacore = datacore
        self.study_features = study_features

        # Optional configuration
        self.feature_limits = feature_limits
        self.feature_values = feature_values
        self.locality_ranges = locality_ranges
        self.kernel = kernel

        # Set default configuration if required
        if use_default:
            self.set_default_configuration(
                override_existing=False,
                bins=bins,
                strict_limits=strict_limits,
                locality_size=locality_size
            )


    ############################################################
    # Default configuration setters

    def set_default_configuration(
            self,
            override_existing: bool = False,
            bins: int = 50,
            strict_limits: bool = True,
            locality_size: dict[float] = None,
    ) -> None:
        """
        Set default configuration values for the explanation.

        Args:
            override_existing (bool): If True, override existing configuration values.
        """

        if override_existing:
            self.feature_limits = None
            self.feature_values = None
            self.locality_ranges = None
            self.kernel = None

        # FEATURE LIMITS
        if self.feature_limits is None:
            self.feature_limits = self.default_feature_limits(
                bins=bins,
                strict_limits=strict_limits
            )

        # FEATURE VALUES
        if self.feature_values is None:
            self.feature_values = self.default_feature_values(bins=bins)

        # LOCALITY RANGES
        if self.locality_ranges is None:
            self.locality_ranges = self.default_locality_ranges(
                locality_size=locality_size
            )

        # KERNEL
        if self.kernel is None:
            self.kernel = self.defaul_kernel()


    def default_feature_limits(
            self,
            bins: int = 50,
            strict_limits: bool = True) -> dict[str, tuple[float, float]]:
        """
        Get default values for the limits for non-study features.

        If strict_limits is True, the feature limits will be set to the min and max values of each feature in
        the dataset.
        Otherwise, the limits will overpass the dataset limits by 0.5 the size of each bin.
        """

        feature_limits = {}

        # If feature_values are already set, set the limits accordingly
        if self.feature_values is not None:
            for feature in self.study_features:
                values = self.feature_values[feature]
                feature_limits[feature] = (values.min(), values.max())
            return feature_limits

        for feature in self.study_features:
            feature_limit = (
                self.datacore.df_X[feature].min(),
                self.datacore.df_X[feature].max()
            )

            if not strict_limits:
                # Calculate the bin width
                total_range = feature_limit[1] - feature_limit[0]
                bin_width = total_range / bins

                # Extend the limits by half a bin width
                feature_limit = (
                    feature_limit[0] - bin_width / 2,
                    feature_limit[1] + bin_width / 2
                )
            feature_limits[feature] = feature_limit

        return feature_limits


    def default_feature_values(
            self,
            bins: int = 50
    ) -> np.ndarray:
        """
        Set default values for the feature values for non-study features.

        The feature values will be set as evenly spaced values within the feature limits.
        """

        feature_values = {}

        for feature in self.study_features:
            limits = self.feature_limits[feature]
            feature_values[feature] = np.linspace(
                limits[0],
                limits[1],
                bins
            )

        return feature_values


    def default_locality_ranges(
            self,
            locality_size: dict[float] = None,
    ) -> dict[np.ndarray]:
        """
        Set default locality ranges for the non-study features.

        The locality ranges will be set depending on the arguments.
        If locality_size is provided, it will be used as the locality range for all features, starting from
        the feature limits minimum value.
        Otherwise, the standard deviation of each feature will be used as the locality range.
        """

        locality_ranges = {}

        for feature in self.study_features:

            # If no locality set, use standard deviation
            if locality_size is not None and feature in locality_size:
                locality = locality_size[feature]

            else:
                locality = self.datacore.df_X[feature].std()

            # Generate an array of float from limits minimum of the feature with separation of locality
            limits = self.feature_limits[feature]
            locality_ranges[feature] = np.arange(
                limits[0],
                limits[1],
                locality
            )

        return locality_ranges


    def defaul_kernel(
            self
    ) -> Kernel:
        """
        Set default kernel for the explanation.

        The kernel will be instantiated using the provided kernel constructor.
        """
        return create_default_kernel(self.datacore.df_X[self.datacore.features()])


    ############################################################
    # Data generation functions

    @cache_method
    def get_grid(self) -> Grid:
        """
        Generate a grid with the feature values for the study features.
        """
        return Grid([self.feature_values[feature] for feature in self.study_features])


    @cache_method
    def get_grid_dataframe(self) -> pd.DataFrame:
        """
        Generate a grid DataFrame where study features create a grid over the feature values.
        For each feature value combination, all non-study features are repeated with their feature values.
        """

        # Build the Cartesian product of all grid axes as a DataFrame
        prod = pd.MultiIndex.from_product(
            [np.asarray(self.feature_values[c]) for c in self.study_features], names=self.study_features
        ).to_frame(index=False)

        # Cross join: repeat each df row (restricted to non-grid cols) for every grid combo
        left = prod.assign(_k=1)
        right = self.datacore.df_X[self.non_study_features()].assign(_k=1) if self.non_study_features() else pd.DataFrame({'_k':[1]})
        out = left.merge(right, on="_k").drop(columns="_k")

        return out


    ############################################################
    # Auxiliary functions

    @cache_method
    def non_study_features(self) -> list[str]:
        """Get the name of the non-study features."""
        return [
            feature for feature in self.datacore.features()
            if feature not in self.study_features
        ]


    ############################################################
    # Check functions and utilities

    def check(self, throw: bool = True) -> bool:
        """
        Validate the configuration parameters.

        Args:
            throw (bool): If True, raise an exception if the configuration is invalid.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """

        # Check core configuration
        if self.datacore is None:
            if throw:
                raise ValueError("Core configuration is not set.")
            return False

        # Check study features
        if not self.study_features:
            if throw:
                raise ValueError("Study features are not set.")
            return False

        else:
            for feature in self.study_features:
                if feature not in self.datacore.features():
                    if throw:
                        raise ValueError(f"Study feature '{feature}' is not in the core configuration features.")
                    return False

        return True

    def check_kernel(self, throw: bool = True) -> bool:
        """
        Validate the kernel configuration.

        Args:
            throw (bool): If True, raise an exception if the kernel configuration is invalid.

        Returns:
            bool: True if the kernel configuration is valid, False otherwise.
        """

        if self.kernel is None:
            if throw:
                raise ValueError("Kernel configuration is not set.")
            return False

        return True

    def check_locality_ranges(self, throw: bool = True) -> bool:
        """
        Validate the locality ranges configuration.

        Args:
            throw (bool): If True, raise an exception if the locality ranges configuration is invalid.

        Returns:
            bool: True if the locality ranges configuration is valid, False otherwise.
        """

        if self.locality_ranges is None:
            if throw:
                raise ValueError("Locality ranges configuration is not set.")
            return False

        return True
