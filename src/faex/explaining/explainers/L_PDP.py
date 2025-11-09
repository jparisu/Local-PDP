"""
Local Partial Dependence Plot (l-PDP) Explainer.
This class holds the data and methods for generating l-PDP distributions and plots.
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


class L_PDP(CacheExplainerData, ExplainerPlot):
    def check_configuration(cls, configuration: ExplainerConfiguration, throw: bool = True) -> bool:
        valid = True

        # Check the datacore, features and feature values
        valid = valid and configuration.check(throw=throw)

        # Check for kernel
        valid = valid and configuration.check_kernel(throw=throw)

        return valid

    def _explain(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        context: ExplainerContext,
    ) -> HyperPlane:
        logger.debug("l-PDP explanation generation")

        # Get ICE values
        # Ice holds the grid and the already predicted values
        lice: HyperPlanes = context.explain("l-ice")
        grid = lice.grid
        values = lice.targets
        weights = lice.weights

        # Get the normalization weight
        kernel_normalizer: HyperPlane = context.explain("kernel-normalizer")

        # Get the predictions by averaging ICE values with weights across all instances
        weithted = values * weights
        predictions = weithted.mean(axis=0) / kernel_normalizer.target

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
        Plot the l-PDP values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """
        params = dict(params) if params else {}

        params.setdefault("color", "darkblue")
        params.setdefault("label", "l-PDP")
        params.setdefault("linewidth", 3)

        hyperplane = self.explain(context)

        return from_hyperplane_to_line(
            hyperplane=hyperplane,
            params=params,
        )
