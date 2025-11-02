"""
Individual Conditional Expectation (ICE) class.
This class holds the data and methods for generating ICE distributions and plots.
"""

from __future__ import annotations
from typing import Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

from faxai.data.DataHolder import HyperPlane
from faxai.explaining.ExplainerCore import ExplainerCore
from faxai.explaining.configuration.DataCore import DataCore
from faxai.explaining.configuration.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.ExplanationTechnique import ExplanationTechnique

logger = logging.getLogger(__name__)

class ICE(ExplanationTechnique):

    def _explain(
        self,
        configuration: ExplainerConfiguration,
        core: ExplainerCore | None = None,
    ) -> HyperPlane:
        """
        Internal method to generate explanations based on the provided configuration.
        It can use datacore if provided for efficient data handling.

        Args:
            configuration (ExplainerConfiguration): The configuration for the explanation.
            datacore (DataCore): The core data and model information.

        Returns:
            ExplanationTechnique: An instance of the explanation technique with computed explanations.
        """

        logger.debug(f"ICE explanation generation")

        # Get the grid dataframe from configuration
        grid = configuration.get_grid()

        logger.debug(f"ICE grid shape: {grid.shape()}")

        # Get Dataframe to predict from configuration
        to_predict = configuration.get_grid_dataframe()

        logger.debug(f"ICE grid to predict dataframe size: {to_predict.shape}")

        # Get the predictions for every point in the grid
        predictions = configuration.datacore.predict(to_predict)

        # Reshape the predictions to match the grid shape
        n = len(configuration.datacore)
        reshaped_predictions = predictions.reshape(*grid.shape(), n)

        return HyperPlane(grid=grid, target=reshaped_predictions)


    def check_configuration(
            cls,
            configuration: ExplainerConfiguration,
            throw: bool = True
    ) -> bool:
        """
        Check if the provided configuration is valid for this explanation technique.

        It requires:
        - datacore
        - feature study
        - feature_values
        """
        # Check the datacore is provided
        if configuration.datacore is None:
            if throw:
                raise ValueError("DataCore must be provided in the configuration for ICE explanation.")
            return False

        # Check the feature study is provided
        if configuration.study_features is None:
            if throw:
                raise ValueError("Feature study must be provided in the configuration for ICE explanation.")
            return False

        # Check the feature values are provided
        if configuration.feature_values is None:
            if throw:
                raise ValueError("Feature values must be provided in the DataCore for ICE explanation.")
            return False

        return True
