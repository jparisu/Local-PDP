"""
Local - Individual Conditional Expectation (l-ICE) class.
This class holds the data and methods for generating l-ICE distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faex.data.DataHolder import HyperPlanes, WeightedHyperPlanes
from faex.plotting.DataPlotter import DataPlotter
from faex.data.holder_to_plotter import from_weighted_hyperplanes_to_lines
from faex.core.DataCore import DataCore
from faex.explaining.Explainer import ExplainerPlot
from faex.core.DataCore import DataCore
from faex.explaining.explainers.CacheExplainer import CacheExplainerData
from faex.explaining.ExplainerFactory import ExplainerFactory

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class L_ICE(CacheExplainerData, ExplainerPlot):
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
    ) -> WeightedHyperPlanes:
        logger.debug("l-ICE explanation generation")

        # Get the ICE information
        ice: HyperPlanes = context.explain("ice")

        # For each point in the target, calculate the kernel weights
        kernel_values: HyperPlanes = context.explain("kernel-values")

        # Create the weighted hyperplanes
        return WeightedHyperPlanes(
            grid=ice.grid,
            targets=ice.targets,
            weights=kernel_values.targets,
        )

    def plot(self, context: ExplainerContext, params: dict = None) -> DataPlotter:
        """
        Plot the l-ICE values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """
        params = dict(params) if params else {}

        params.setdefault("color", "lime")
        params.setdefault("label", "l-ICE")
        params.setdefault("linewidth", 1)
        params.setdefault("opacity", 0.2)

        hyperplane = self.explain(context)

        return from_weighted_hyperplanes_to_lines(
            w_hyperplanes=hyperplane,
            params=params,
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=L_ICE,
    aliases=["l-ice", "local-ice"],
)
