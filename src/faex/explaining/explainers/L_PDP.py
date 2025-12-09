"""
Local Partial Dependence Plot (l-PDP) Explainer.
This class holds the data and methods for generating l-PDP distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faex.data.DataHolder import WeightedHyperPlane, WeightedHyperPlanes, DistributionCollection
from faex.plotting.DataPlotter import DataPlotter
from faex.plotting.plotter_params import AreaParams, ErrorBarParams
from faex.data.holder_to_plotter import from_weighted_hyperplane_to_line, from_distributions_to_area, from_distribution_to_line_with_error_bar
from faex.explaining.Explainer import ExplainerPlot
from faex.core.DataCore import DataCore
from faex.explaining.explainers.CacheExplainer import CacheExplainerData
from faex.explaining.ExplainerFactory import ExplainerFactory
from faex.plotting.d2.dataplotter_special_2d import DP2_WeightedWidthLine
from faex.mathing.distribution.sampling_distributions import DeltaWeightedDistribution

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class L_PDP(CacheExplainerData, ExplainerPlot):
    def check_configuration(cls, configuration: DataCore, throw: bool = True) -> bool:
        valid = True

        # Check the datacore, features and feature values
        valid = valid and configuration.check(throw=throw)

        # Check for kernel
        valid = valid and configuration.check_kernel(throw=throw)

        return valid

    def _explain(
        self,
        context: ExplainerContext,
    ) -> WeightedHyperPlane:
        logger.debug("l-PDP explanation generation")

        # Get ICE values
        # Ice holds the grid and the already predicted values
        lice: WeightedHyperPlanes = context.explain("l-ice")
        grid = lice.grid
        values = lice.targets
        weights = lice.weights

        # Get the normalization weight
        kernel_normalizer: WeightedHyperPlane = context.explain("kernel-normalizer")

        # Get the predictions by averaging ICE values with weights across all instances
        weithted = values * weights
        predictions = weithted.sum(axis=0) / kernel_normalizer.target

        # Reshape the predictions to match the grid shape

        return WeightedHyperPlane(
            grid=grid,
            target=predictions,
            weights=kernel_normalizer.target,
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

        params.setdefault("color", "darkgreen")
        params.setdefault("label", "l-PDP")
        params.setdefault("linewidth", 3)
        params.setdefault("opacity", 1.0)

        hyperplane = self.explain(context)

        logger.debug(f"Plotting l-PDP with params: {params}")

        return from_weighted_hyperplane_to_line(
            w_hyperplane=hyperplane,
            params=params,
            type_of_line=DP2_WeightedWidthLine
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=L_PDP,
    aliases=["l-pdp", "local-pdp"]
)


class L_PDP_Distribution(CacheExplainerData, ExplainerPlot):

    def _explain(
        self,
        context: ExplainerContext,
    ) -> DistributionCollection:

        logger.debug("PDP distribution explanation generation")

        # Get ICE values
        lice: WeightedHyperPlanes = context.explain("lice")
        return lice.to_distributions(max_weight=context.configuration.kernel.maximum())


    def plot(
        self,
        context: ExplainerContext,
        params: dict = None,
    ) -> DataPlotter:
        """
        Plot the PDP distribution values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """

        logger.debug("PDP distribution visualization generation")

        params = AreaParams.from_dict(params)

        params.set_default_value("color", "green")
        params.set_default_value("label", "l-PDP-Distribution")
        params.set_default_value("alpha", 0.5)

        distribution_collection = context.explain(self.name())

        return from_distributions_to_area(
            distributions=distribution_collection,
            params=params,
        )


# Register Explainer
ExplainerFactory.register_explainer(
    explainer=L_PDP_Distribution,
    aliases=["l-pdp-d"]
)



class L_PDP_Error(ExplainerPlot):

    def plot(
        self,
        context: ExplainerContext,
        params: dict = None,
    ) -> DataPlotter:
        """
        Plot the PDP distribution values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """

        logger.debug("PDP distribution visualization generation")

        params = ErrorBarParams.from_dict(params)

        params.set_default_value("color", "darkgreen")
        params.set_default_value("label", "l-PDP-Error")
        params.set_default_value("alpha", 0.9)

        distribution_collection = context.explain("lpdpdistribution")

        return from_distribution_to_line_with_error_bar(
            distributions=distribution_collection,
            params=params,
        )


# Register Explainer
ExplainerFactory.register_explainer(
    explainer=L_PDP_Error,
    aliases=["l-pdp-e"]
)
