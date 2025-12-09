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
from faex.core.DataCore import DataCore
from faex.classifying.ClassifierModelHandler import ClassifierModelHandler

logger = logging.getLogger(__name__)


class ClassificationDataCore(DataCore):
    """
    Configuration class for explanation generation.

    This class holds various parameters that can be adjusted to customize different explanation techniques.
    This facilitates flexibility and adaptability in generating explanations for machine learning models.
    """

    def __init__(
        self,

        # Required configuration
        class_label: str,
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
        self.class_label = class_label

        # Modify the model to predict the probability of the given class label
        model = ClassifierModelHandler(model, class_label)

        super().__init__(
            model=model,
            df_X=df_X,
            study_features=study_features,
            feature_limits=feature_limits,
            feature_values=feature_values,
            locality_limits=locality_limits,
            kernel=kernel,
            use_default=use_default,
            bins=bins,
            strict_limits=strict_limits,
            locality_size=locality_size,
            sigma_factor=sigma_factor,
            locality_factor=locality_factor,
            rng=rng,
            data_percentage=data_percentage,
            max_samples=max_samples,
        )
