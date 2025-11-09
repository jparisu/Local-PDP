"""
Marginal Partial Dependence Plot (M-PDP) Explainer.
This class holds the data and methods for generating M-PDP distributions and plots.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faxai.data.DataHolder import HyperPlane, HyperPlanes, DataHolderCollection
from faxai.data.DataPlotter import DataPlotter
from faxai.data.holder_to_plotter import from_collection_to_lines
from faxai.explaining.DataCore import DataCore
from faxai.explaining.Explainer import ExplainerPlot
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.explainers.CacheExplainer import CacheExplainerData

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faxai.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class M_PDP(CacheExplainerData, ExplainerPlot):
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

        logger.debug("M-PDP explanation generation")

        # Get ICE values
        # Ice holds the grid and the already predicted values
        mice: DataHolderCollection = context.explain("m-ice")

        holder = DataHolderCollection()

        # For each Hyperplanes in mice, we need to average the targets
        for locality in mice:
            locality : HyperPlanes

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
        params = dict(params) if params else {}

        # TODO: mix grids to create a single line

        params.setdefault("color", "darkblue")
        params.setdefault("label", "m-PDP")
        params.setdefault("linewidth", 3)

        collection = self.explain(context)

        return from_collection_to_lines(
            collection=collection,
            params=params,
        )
