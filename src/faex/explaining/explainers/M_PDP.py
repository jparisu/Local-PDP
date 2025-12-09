"""
Marginal Partial Dependence Plot (M-PDP) Explainer.
This class holds the data and methods for generating M-PDP distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faex.data.DataHolder import DataHolderCollection, HyperPlane, HyperPlanes
from faex.plotting.DataPlotter import DataPlotter
from faex.data.holder_to_plotter import from_collection_to_lines
from faex.core.DataCore import DataCore
from faex.explaining.Explainer import ExplainerPlot
from faex.explaining.explainers.CacheExplainer import CacheExplainerData
from faex.explaining.ExplainerFactory import ExplainerFactory

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class M_PDP(CacheExplainerData, ExplainerPlot):
    def check_configuration(cls, configuration: DataCore, throw: bool = True) -> bool:
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
        context: ExplainerContext,
    ) -> HyperPlane:
        logger.debug("M-PDP explanation generation")

        # Get ICE values
        # Ice holds the grid and the already predicted values
        mice: DataHolderCollection = context.explain("m-ice")

        holder = DataHolderCollection()

        # For each Hyperplanes in mice, we need to average the targets
        for locality in mice:
            locality: HyperPlanes

            grid = locality.grid
            targets = locality.targets

            # Create a hyperPlane for the PDP in this locality
            predictions = targets.mean(axis=0)
            holder.add(
                HyperPlane(
                    grid=grid,
                    target=predictions,
                )
            )

        return holder

    def plot(
        self,
        context: ExplainerContext,
        params: dict = None,
    ) -> DataPlotter:
        """
        Plot the M-PDP values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """

        logger.debug("m-PDP visualization generation")

        params = dict(params) if params else {}

        # TODO: mix grids to create a single line

        params.setdefault("color", "brown")
        params.setdefault("label", "m-PDP")
        params.setdefault("linewidth", 3)

        collection = context.explain("m-pdp")

        return from_collection_to_lines(
            collection=collection,
            params=params,
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=M_PDP,
    aliases=["m-pdp", "marginal-pdp"]
)
