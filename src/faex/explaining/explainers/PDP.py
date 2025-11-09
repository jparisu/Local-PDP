"""
Partial Dependence Plot (PDP) Explainer.
This class holds the data and methods for generating PDP distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faex.data.DataHolder import HyperPlane, HyperPlanes
from faex.data.DataPlotter import DataPlotter
from faex.data.holder_to_plotter import from_hyperplane_to_line
from faex.explaining.DataCore import DataCore
from faex.explaining.Explainer import ExplainerPlot
from faex.explaining.ExplainerConfiguration import ExplainerConfiguration
from faex.explaining.explainers.CacheExplainer import CacheExplainerData

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class PDP(CacheExplainerData, ExplainerPlot):
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
    ) -> HyperPlane:
        logger.debug("PDP explanation generation")

        # Get ICE values
        # Ice holds the grid and the already predicted values
        ice: HyperPlanes = context.explain("ice")
        grid = ice.grid

        logger.debug(f"PDP grid shape: {grid.shape()}")

        # Get the predictions by averaging ICE values across all instances
        ice_values = ice.targets
        predictions = ice_values.mean(axis=0)

        # Reshape the predictions to match the grid shape

        return HyperPlane(
            grid=grid,
            target=predictions,
        )

    def plot(
        self,
        context: ExplainerContext,
        params: dict = None,
    ) -> DataPlotter:
        """
        Plot the PDP values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """
        params = dict(params) if params else {}

        params.setdefault("color", "darkblue")
        params.setdefault("label", "PDP")
        params.setdefault("linewidth", 3)

        hyperplane = self.explain(context)

        return from_hyperplane_to_line(
            hyperplane=hyperplane,
            params=params,
        )
