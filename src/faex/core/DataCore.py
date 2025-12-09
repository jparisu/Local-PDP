"""
A explanation configuration class allows to set various parameters for generating explanations for
different techniques.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from typing import Any

from faex.data.DataHolder import Grid
from faex.mathing.kernel import Kernel, create_default_kernel, GaussianKernel
from faex.mathing.bandwidth import Bandwidth
from faex.mathing.mathing import reckon_silverman_bandwidth
from faex.utils.decorators import cache_method
from faex.mathing.RandomGenerator import RandomGenerator

logger = logging.getLogger(__name__)


@dataclass
class DataCore:
    """
    Configuration class for explanation generation.

    This class holds various parameters that can be adjusted to customize different explanation techniques.
    This facilitates flexibility and adaptability in generating explanations for machine learning models.
    """

    # Model & Dataset
    model: Any
    df_X: pd.DataFrame

    # Features to study configuration
    study_features: list[str] = None
    feature_limits: dict[str, tuple[float, float]] = None
    feature_values: dict[str, np.ndarray] = None

    # Locality configuration
    locality_limits: dict[str, np.ndarray] = None
    kernel: Kernel = None

    def __init__(
        self,

        # Required configuration
        model: Any,
        df_X: pd.DataFrame,
        study_features: list[str],
        *,

        # Optional configuration
        feature_limits: dict[str, tuple[float, float]] | None = None,
        feature_values: dict[str, np.ndarray] | None = None,
        locality_limits: dict[str, np.ndarray] | None = None,
        kernel: Kernel = None,

        # Default configuration arguments
        use_default: bool = True,
        bins: int = 50,
        strict_limits: bool = True,
        locality_size: dict[float] = None,
        sigma_factor: float = None,
        locality_factor: float = None,

        # Random selection
        rng: RandomGenerator = RandomGenerator(42),
        data_percentage: float = None,
        max_samples: int = None,
    ):
        """
        Initialize the ExplanationConfiguration with the core and study features.

        Args:
            model (Any): The machine learning model to explain. Must implement function predict().
            df_X (pd.DataFrame): DataFrame containing the data to explain.
            study_features (list[str]): List of feature names to study.
        """

        if data_percentage is not None and data_percentage != 1.0:
            df_X = df_X.sample(frac=data_percentage, random_state=rng.randint(0,10000)).reset_index(drop=True)
        if max_samples is not None and len(df_X) > max_samples:
            df_X = df_X.sample(n=max_samples, random_state=rng.randint(0,10000)).reset_index(drop=True)

        # Input stored
        self.model = model
        self.df_X = df_X
        self.study_features = study_features

        # Optional configuration
        self.feature_limits = feature_limits
        self.feature_values = feature_values
        self.locality_limits = locality_limits
        self.kernel = kernel

        # Set default configuration if required
        if use_default:
            self.set_default_configuration(
                override_existing=False,
                bins=bins,
                strict_limits=strict_limits,
                locality_size=locality_size,
                sigma_factor=sigma_factor,
                locality_factor=locality_factor,
            )

    ############################################################
    # Model and Dataset functions

    def features(self) -> list[str]:
        """Get the feature names from the DataFrame."""
        return list(self.df_X.columns)

    @cache_method
    def get_real_predictions(self) -> np.ndarray:
        """Get the real predictions from the model."""
        return self.model.predict(self.df_X)

    def predict(self, df_X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target using the model for the given data.

        Args:
            df_X (pd.DataFrame): DataFrame containing the data to predict.

        Returns:
            np.ndarray: Predictions from the model.

        Note:
            It uses a weak reference to re-use predictions for data-frames already predicted.
        """

        # Check if predictions are cached
        # TODO: TypeError: unhashable type: 'DataFrame'
        # if df_X in self.__cache_predictions_df:
        #     logger.debug(f"Using cached predictions in {self} for the provided data {df_X}.")
        #     return self.__cache_predictions_df[df_X]

        # Predict
        predictions = self.model.predict(df_X)

        # Check predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        # Store in cache
        # logger.debug(f"Storing predictions in cache in {self} for the provided data {df_X}.")
        # self.__cache_predictions_df[df_X] = predictions

        return predictions


    ############################################################
    # Default configuration setters

    def set_default_configuration(
        self,
        override_existing: bool,
        bins: int = 50,
        strict_limits: bool = True,
        locality_size: dict[float] = None,
        sigma_factor: float = None,
        locality_factor: float = None,
    ) -> None:
        """
        Set default configuration values for the explanation.

        Args:
            override_existing (bool): If True, override existing configuration values.
        """

        if override_existing:
            self.feature_limits = None
            self.feature_values = None
            self.locality_limits = None
            self.kernel = None

        # FEATURE LIMITS
        if self.feature_limits is None:
            self.feature_limits = self.default_feature_limits(bins=bins, strict_limits=strict_limits)

        # FEATURE VALUES
        if self.feature_values is None:
            self.feature_values = self.default_feature_values(bins=bins)

        # LOCALITY RANGES
        if self.locality_limits is None:
            self.locality_limits = self.default_locality_limits(
                locality_size=locality_size,
                sigma_factor=sigma_factor,
                locality_factor=locality_factor,)

        # KERNEL
        if self.kernel is None:
            self.kernel = self.default_kernel(
                sigma_factor=sigma_factor,
                locality_factor=locality_factor,)

    def default_feature_limits(self, bins: int = 50, strict_limits: bool = True) -> dict[str, tuple[float, float]]:
        """
        Get default values for the limits for non-study features.

        If strict_limits is True, the feature limits will be set to the min and max values of each feature in
        the dataset.
        Otherwise, the limits will overpass the dataset limits by None the size of each bin.
        """

        feature_limits = {}

        # If feature_values are already set, set the limits accordingly
        if self.feature_values is not None:
            for feature in self.study_features:
                values = self.feature_values[feature]
                feature_limits[feature] = (values.min(), values.max())
            return feature_limits

        for feature in self.study_features:
            feature_limit = (self.df_X[feature].min(), self.df_X[feature].max())

            if not strict_limits:
                # Calculate the bin width
                total_range = feature_limit[1] - feature_limit[0]
                bin_width = total_range / bins

                # Extend the limits by half a bin width
                feature_limit = (feature_limit[0] - bin_width / 2, feature_limit[1] + bin_width / 2)
            feature_limits[feature] = feature_limit

        return feature_limits

    def default_feature_values(self, bins: int = 50) -> np.ndarray:
        """
        Set default values for the feature values for non-study features.

        The feature values will be set as evenly spaced values within the feature limits.
        """

        feature_values = {}

        for feature in self.study_features:
            limits = self.feature_limits[feature]
            feature_values[feature] = np.linspace(limits[0], limits[1], bins)

        return feature_values

    def default_locality_limits(
        self,
        locality_size: dict[float] = None,
        sigma_factor: float = None,
        locality_factor: float = None,
    ) -> dict[str, np.ndarray]:
        """
        Set default locality ranges for the non-study features.

        The locality ranges will be set depending on the arguments.
        If locality_size is provided, it will be used as the locality range for all features, starting from
        the feature limits minimum value.
        Otherwise, the standard deviation of each feature will be used as the locality range.
        """

        locality_limits = {}

        for feature in self.study_features:
            # If no locality set, use standard deviation
            if locality_size is not None and feature in locality_size:
                locality = locality_size[feature]

            else:
                # If sigma factor provided, calculate locality using sigma
                sigma = self.df_X[feature].std()

                if sigma_factor is not None:
                    locality = sigma * sigma_factor
                else:
                    locality = reckon_silverman_bandwidth(len(self.df_X), sigma)
                    if locality_factor:
                        locality *= locality_factor

            # Generate an array of float from limits minimum of the feature with separation of locality
            limits = self.feature_limits[feature]
            locality_limits[feature] = np.arange(limits[0], limits[1], locality)

            locality_limits[feature] = np.append(locality_limits[feature], np.inf)

            # Set first value to -inf
            locality_limits[feature][0] = -np.inf

        return locality_limits

    def default_kernel(
            self,
            sigma_factor: float = None,
            locality_factor: float = None,
    ) -> Kernel:
        """
        Set default kernel for the explanation.

        The kernel will be instantiated using the provided kernel constructor.
        """
        if sigma_factor is not None:
            bandwidth = Bandwidth.build_diagonal(
                self.df_X[self.study_features].std().to_numpy() * sigma_factor,
            )

        elif locality_factor is not None:
            bandwidth = Bandwidth.reckon_silverman_bandwidth_from_data(self.df_X[self.study_features])
            bandwidth = bandwidth.scale(locality_factor)

        else:
            return create_default_kernel(self.df_X[self.study_features])

        return GaussianKernel(bandwidth=bandwidth)

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
        right = (
            self.df_X[self.non_study_features()].assign(_k=1)
            if self.non_study_features()
            else pd.DataFrame({"_k": [1]})
        )
        out = left.merge(right, on="_k").drop(columns="_k")

        # Rearrange columns to match datacore
        out = out[self.df_X.columns]

        return out

    ############################################################
    # Auxiliary functions

    @cache_method
    def non_study_features(self) -> list[str]:
        """Get the name of the non-study features."""
        return [feature for feature in self.features() if feature not in self.study_features]

    def locality_ranges(self) -> dict[str, list[tuple[float, float]]]:
        """
        Get the locality ranges as a dictionary of feature name to list of (min, max) tuples.

        Returns:
            dict[str, list[tuple[float, float]]]: Locality ranges for each study feature.
        """

        locality_ranges = {}

        for feature in self.study_features:
            limits = self.feature_limits[feature]
            localities = self.locality_limits[feature]

            ranges = []
            for i in range(1, len(localities)):
                range_min = localities[i - 1]
                range_max = localities[i]
                ranges.append((range_min, range_max))

            locality_ranges[feature] = ranges

        return locality_ranges

    def study_feature_dataframe(self) -> pd.DataFrame:
        """
        Get the feature values for the study features as a 2D numpy array.

        Returns:
            np.ndarray: 2D array of shape (num_study_features, num_feature_values).
        """
        return self.df_X[self.study_features]

    def __str__(self) -> str:
        """
        String representation of the DataCore.

        Returns:
            str: String representation of the configuration.
        """
        return (
            f"DataCore(\n"
            f"  model={self.model},\n"
            f"  df_X shape={self.df_X.shape},\n"
            f"  study_features={self.study_features},\n"
            f"  feature_limits={self.feature_limits},\n"
            f"  feature_values size={ {k: v.shape for k, v in self.feature_values.items()} },\n"
            f"  locality_limits={self.locality_limits},\n"
            f"  kernel={self.kernel}\n"
            f")"
        )

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

        # Validate model is not None
        if self.model is None:
            if throw:
                raise ValueError("Model cannot be None.")
            return False

        # Validate DataFrame
        if self.df_X is None or not isinstance(self.df_X, pd.DataFrame):
            if throw:
                raise ValueError("df_X must be a valid pandas DataFrame.")
            return False

        # Check the DataFrame is a valid dataset for model.predict method
        try:
            _ = self.model.predict(self.df_X.head(1))
        except Exception as e:
            if throw:
                raise ValueError(f"The model and the dataset are not compatible: {e}") from e
            return False

        # Check study features
        if not self.study_features:
            if throw:
                raise ValueError("Study features are not set.")
            return False

        else:
            for feature in self.study_features:
                if feature not in self.study_features:
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

        if self.locality_limits is None:
            if throw:
                raise ValueError("Locality ranges configuration is not set.")
            return False

        return True


    ############################################################
    # Conversion functions

    def to_univariate(self) -> UnivariateDataCore:
        """
        Convert the current configuration to a UnivariateDataCore.

        Returns:
            UnivariateDataCore: The univariate configuration.
        """
        if len(self.study_features) != 1:
            raise ValueError("Cannot convert to UnivariateDataCore: study_features length is not 1.")

        return UnivariateDataCore(
            model=self.model,
            df_X=self.df_X,
            study_features=self.study_features,
            feature_limits=self.feature_limits,
            feature_values=self.feature_values,
            locality_limits=self.locality_limits,
            kernel=self.kernel,
        )


class UnivariateDataCore(DataCore):
    """
    Configuration class for univariate explanation generation.

    This class holds various parameters that can be adjusted to customize univariate explanation techniques.
    This facilitates flexibility and adaptability in generating explanations for machine learning models.

    Attributes:
        model (Any): The machine learning model to explain. Must implement function predict().
        df_X (pd.DataFrame): The dataset used for generating explanations.
        study_feature (str): The feature name to study.
        feature_limit (tuple[float, float]): The limits for the study feature.
        feature_values (np.ndarray): The values for the study feature.
        locality_limits (np.ndarray): The locality limits for the study feature.
        kernel (Kernel): The kernel to use for locality weighting.
    """

    def uni_study_feature(self):
        return self.study_features[0]

    def uni_feature_values(self):
        return self.feature_values[self.uni_study_feature()]

    def uni_feature_limits(self):
        return self.feature_limits[self.uni_study_feature()]

    def uni_locality_limits(self):
        return self.locality_limits[self.uni_study_feature()]

    def uni_histogram_limits(self):
        # Set an array where first and las are not infinite
        limits = self.uni_locality_limits().copy()
        limits[0] = self.uni_feature_limits()[0]
        limits[-1] = self.uni_feature_limits()[1]
        return limits

    def uni_study_feature_array(self) -> np.ndarray:
        """
        Get the feature values for the study feature as a 1D numpy array.
        Returns:
            np.ndarray: 1D array of shape (num_feature_values,).
        """
        return self.df_X[self.uni_study_feature()].to_numpy()

    def to_univariate(self):
        return self
