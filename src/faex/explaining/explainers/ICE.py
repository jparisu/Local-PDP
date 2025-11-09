"""
Individual Conditional Expectation (ICE) class.
This class holds the data and methods for generating ICE distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faex.data.DataHolder import HyperPlanes
from faex.data.DataPlotter import DataPlotter
from faex.data.holder_to_plotter import from_hyperplanes_to_lines, from_hyperplanes_to_scatter
from faex.explaining.DataCore import DataCore
from faex.explaining.Explainer import ExplainerPlot
from faex.explaining.ExplainerConfiguration import ExplainerConfiguration
from faex.explaining.explainers.CacheExplainer import CacheExplainerData

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class ICE(CacheExplainerData, ExplainerPlot):
    def check_configuration(cls, configuration: ExplainerConfiguration, throw: bool = True) -> bool:
        """
        Check if the provided configuration is valid for this explanation technique.

        It requires:
        - datacore
        - feature study
        - feature_values
        """
        valid = True

        # Check the datacore, features and feature values
        valid = valid and configuration.check(throw=throw)

        return valid

    def _explain(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        context: ExplainerContext,
    ) -> HyperPlanes:
        logger.debug("ICE explanation generation")

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
        new_shape = grid.shape() + (n,)
        reshaped_predictions = predictions.reshape(new_shape)

        # Transpose to have shape (n, grid_size)
        reshaped_predictions = reshaped_predictions.transpose(-1, *range(len(grid.shape())))

        return HyperPlanes(grid=grid, targets=reshaped_predictions)

    def plot(self, context: ExplainerContext, params: dict = None) -> DataPlotter:
        """
        Plot the ICE values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """
        params = dict(params) if params else {}

        params.setdefault("color", "paleturquoise")
        params.setdefault("label", "ICE")
        params.setdefault("linewidth", 1)
        params.setdefault("alpha", 0.2)

        hyperplane = self.explain(context)

        return from_hyperplanes_to_lines(
            hyperplanes=hyperplane,
            params=params,
        )


class ICE_Scatter(ExplainerPlot):
    def plot(self, context: ExplainerContext, params: dict = None) -> DataPlotter:
        """
        Plot the ICE scatter values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """
        params = dict(params) if params else {}

        params.setdefault("color", "teal")
        params.setdefault("label", "ICE Scatter")
        params.setdefault("s", 10)
        params.setdefault("alpha", 0.5)

        hyperplane = context.explain("ice")

        return from_hyperplanes_to_scatter(
            hyperplanes=hyperplane,
            params=params,
        )
