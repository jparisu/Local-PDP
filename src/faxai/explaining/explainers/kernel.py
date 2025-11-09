"""
Submodule to handle kernel information regarding the ICE and the data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np
from itertools import product

from faxai.data.DataPlotter import DP_Scatter, DP_Histogram
from faxai.data.DataHolder import HyperPlanes, HyperPlane
from faxai.data.holder_to_plotter import from_hyperplanes_to_lines
from faxai.explaining.DataCore import DataCore
from faxai.explaining.Explainer import ExplainerPlot, ExplainerData
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.explainers.CacheExplainer import CacheExplainerData

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faxai.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class KernelValues(CacheExplainerData):
    """
    Represent the kernel weight for each of the ICE points.
    """

    def check_configuration(cls, configuration: ExplainerConfiguration, throw: bool = True) -> bool:
        valid = True

        # Check the datacore, features and feature values
        valid = valid and configuration.check(throw=throw)

        # Check for kernel
        valid = valid and configuration.check_kernel(throw=throw)

        return valid


    @classmethod
    def name(cls) -> str:
        return "kernel-values"


    def _explain(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        context: ExplainerContext,
     ) -> HyperPlanes:

        logger.debug("Calculating kernel values")

        kernel = configuration.kernel
        feature_values = configuration.feature_values
        df = datacore.df_X
        features = configuration.study_features
        n = len(df)
        instance_values = datacore.df_X[features].to_numpy()

        # Get ICE values
        ice : HyperPlanes = context.explain("ice")

        # To store the kernel values
        k_values = np.zeros_like(ice.targets)

        # Create an index that pass through every grid possibility in feature values
        indexer = [0 for _ in range(len(features))]

        finished_indexing = False

        while not finished_indexing:

            # Get the current feature values from the indexer
            current_values = np.array([feature_values[features[i]][indexer[i]] for i in range(len(features))])

            logger.debug(f"Calculating kernel values for indexes {indexer}  with grid point: {current_values}")
            logger.debug(f"Instance: {instance_values[0].shape}, Current Values: {current_values.shape}, Index: {(0,) + tuple(indexer)}")
            logger.debug(f"Instance: {instance_values[0]}, Current Values: {current_values}")

            # Calculate the kernel values for every point in the data
            for i, instance in enumerate(instance_values):
                index = (i,) + tuple(indexer)
                k_values[index] = kernel.apply(instance, current_values)

            ######
            # Update the indexer
            index_updater = len(features) - 1
            indexer[index_updater] += 1

            while indexer[index_updater] >= len(feature_values[features[index_updater]]):
                indexer[index_updater] = 0
                index_updater -= 1
                indexer[index_updater] += 1

                if index_updater < 0:
                    finished_indexing = True
                    break

        return HyperPlanes(
            grid=ice.grid,
            targets=k_values
        )



class KernelNormalizer(CacheExplainerData):
    """
    Represent the sum of the kernel weights for each point in the grid
    """

    def check_configuration(cls, configuration: ExplainerConfiguration, throw: bool = True) -> bool:
        valid = True

        # Check the datacore, features and feature values
        valid = valid and configuration.check(throw=throw)

        # Check for kernel
        valid = valid and configuration.check_kernel(throw=throw)

        return valid


    @classmethod
    def name(cls) -> str:
        return "kernel-normalizer"


    def _explain(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        context: ExplainerContext,
     ) -> HyperPlane:

        logger.debug("Calculating kernel normalizer")

        # Get the kernel values
        kernel_values : HyperPlanes = context.explain("kernel-values")

        # Sum the kernel values across all instances
        grid = kernel_values.grid
        values = kernel_values.targets

        normalizer = values.sum(axis=0)

        return HyperPlane(
            grid=grid,
            target=normalizer,
        )
