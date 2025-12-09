"""
Marginal - Individual Conditional Expectation (m-ICE) class.
This class holds the data and methods for generating ICE distributions and plots using locality ranges.
"""

from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np

from faex.data.DataHolder import DataHolderCollection, HyperPlanes
from faex.plotting.DataPlotter import DataPlotter
from faex.data.holder_to_plotter import from_collection_to_lines
from faex.explaining.Explainer import ExplainerPlot
from faex.core.DataCore import DataCore
from faex.explaining.explainers.CacheExplainer import CacheExplainerData
from faex.explaining.ExplainerFactory import ExplainerFactory

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


# TODO : do the marginal and the kernel vecinity in parallel
class M_ICE(CacheExplainerData, ExplainerPlot):
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

        # Check locality
        valid = valid and configuration.check_locality_ranges(throw=throw)

        return valid

    def _explain(
        self,
        context: ExplainerContext,
    ) -> DataHolderCollection:

        logger.debug("m-ICE explanation generation")

        configuration = context.configuration

        # Get the ICE values
        ice: HyperPlanes = context.explain("ice")
        grid = ice.grid
        targets = ice.targets

        # New Data Holder
        holder = DataHolderCollection()

        # Get locality ranges
        locality_ranges = configuration.locality_ranges()

        # Get the features, that must be in the same order as the grid
        features = configuration.study_features

        # Get the actual dataframe
        dataframe = configuration.df_X

        ######
        # For each locality range, generate new HyperPlanes with all the targets inside it

        # Generate the product for every locality range combination
        localities = itertools.product(*[locality_ranges[feature] for feature in features])

        # For each locality combination
        for locality in localities:
            # Create a dictionary for the current locality
            current_locality = {features[j]: locality[j] for j in range(len(features))}

            # Get the indexes of the instances inside the locality
            instances_indexes = []
            for i, instance in dataframe.iterrows():
                inside_locality = True
                for j, feature in enumerate(features):
                    f_value = instance[feature]
                    f_min, f_max = locality[j]
                    if not (f_min <= f_value <= f_max):
                        inside_locality = False
                        break
                if inside_locality:
                    instances_indexes.append(i)

            # Generate new grid from the old grid and this locality
            new_grid = []
            for j, feature in enumerate(features):
                grid_dimension = grid[j]

                # Get the indexes within the locality range
                f_min, f_max = current_locality[feature]
                ind = grid_dimension >= f_min
                ind = ind & (grid_dimension <= f_max)

                new_grid_dimension = grid_dimension[ind]
                new_grid.append(new_grid_dimension)

            # For each instance inside the locality, get the corresponding target values
            new_targets = []
            for i in instances_indexes:
                target = targets[i]

                # Filter the target to only those indexes
                indexes = []
                for j, feature in enumerate(features):
                    grid_dimension = grid[j]

                    # Get the indexes within the locality range
                    f_min, f_max = current_locality[feature]
                    ind = grid_dimension >= f_min
                    ind = ind & (grid_dimension <= f_max)

                    indexes.append(ind)

                new_target = target[tuple(indexes)]
                new_targets.append(new_target)

            # Generate new Hyperplanes
            holder.add(
                data_holder=HyperPlanes(
                    grid=new_grid,
                    targets=np.array(new_targets),
                )
            )

        return holder

    # Generate one line per instance
    # def _explain(
    #     self,
    #     datacore: DataCore,
    #     configuration: DataCore,
    #     context: ExplainerContext,
    # ) -> DataHolderCollection:

    #     logger.debug("m-ICE explanation generation")

    #     # Get the ICE values
    #     ice: HyperPlanes = context.explain("ice")
    #     grid = ice.grid
    #     targets = ice.targets

    #     # New Data Holder
    #     holder = DataHolderCollection()

    #     # Get locality ranges
    #     locality_ranges = configuration.locality_ranges

    #     # Get the features, that must be in the same order as the grid
    #     features = configuration.study_features

    #     # Get the actual dataframe
    #     dataframe = datacore.df_X

    #     # For each hyperplane, filter values based on locality ranges
    #     for i, target in enumerate(targets):

    #         # First, find in which locality range the values of instance i are
    #         instance = dataframe.iloc[i]
    #         instance_locality = []
    #         for j, feature in enumerate(features):
    #             f_value = instance[feature]
    #             f_min, f_max = locality_ranges[feature]
    #             for range_min, range_max in locality_ranges[feature]:
    #                 if range_min <= f_value <= range_max:
    #                     instance_locality.append((range_min, range_max))
    #                     break

    #         # Get indexes within locality ranges
    #         indexes = []
    #         for j, feature in enumerate(features):
    #             grid_dimension = grid[j]

    #             # Get the indexes within the locality range
    #             ind = grid_dimension >= instance_locality[j][0]
    #             ind = ind & (grid_dimension <= instance_locality[j][1])

    #             indexes.append(ind)

    #         # Filter the target to only those indexes
    #         new_target = target[tuple(indexes)]

    #         # Get new grid
    #         new_grid = []
    #         for j, feature in enumerate(features):
    #             grid_dimension = grid[j]
    #             new_grid_dimension = grid_dimension[indexes[j]]
    #             new_grid.append(new_grid_dimension)

    #         # Generate new Hyperplane
    #         grid = new_grid

    #         holder.add(
    #             data_holder=HyperPlane(
    #                 grid=grid,
    #                 targets=new_target,
    #             )
    #         )

    #     # For each hyperplane, get only those values where grid is within locality ranges

    #     return holder

    def plot(self, context: ExplainerContext, params: dict = None) -> DataPlotter:
        """
        Plot the ICE values.

        Args:
            params (dict): Parameters for the plot.

        Returns:
            DataPlotter: The plotter object.
        """

        logger.debug("m-ICE visualization generation")

        params = dict(params) if params else {}

        params.setdefault("color", "chocolate")
        params.setdefault("label", "m-ICE")
        params.setdefault("linewidth", 1)
        params.setdefault("opacity", 0.2)

        collection = context.explain("m-ice")

        return from_collection_to_lines(
            collection=collection,
            params=params,
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=M_ICE,
    aliases=["m-ice", "marginal-ice"]
)
