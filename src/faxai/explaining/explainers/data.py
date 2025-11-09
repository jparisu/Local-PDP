"""
Submodule to handle explainers related to the actual data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from faxai.data.DataPlotter import DP_Scatter, DP_Histogram
from faxai.data.holder_to_plotter import from_hyperplanes_to_lines
from faxai.explaining.DataCore import DataCore
from faxai.explaining.Explainer import ExplainerPlot
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.explainers.CacheExplainer import CacheExplainerData

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faxai.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)

class RealPrediction(ExplainerPlot):

    # TODO check study features set and that is 1d

    @classmethod
    def name(cls) -> str:
        return "real-prediction"

    def plot(self, context: ExplainerContext, params: dict = None) -> DP_Scatter:

        params = dict(params) if params else {}

        params.setdefault('color', 'navy')
        params.setdefault('alpha', 0.5)
        params.setdefault('label', 'predicted')
        params.setdefault('s', 5)

        # Get the study feature dataframe
        df = context.configuration.study_feature_dataframe()

        # Check it is only one column
        if df.shape[1] != 1:
            raise ValueError("RealPrediction plot requires a single study feature.")
        x_values = df.iloc[:, 0]

        return DP_Scatter(
            x=x_values,
            y=context.datacore.get_real_predictions(),
            params=params
        )


class Histogram(ExplainerPlot):

    @classmethod
    def name(cls) -> str:
        return "histogram"

    def plot(self, context: ExplainerContext, params: dict = None) -> DP_Scatter:

        params = dict(params) if params else {}

        params.setdefault('density', True)
        params.setdefault('color', 'aquamarine')
        params.setdefault('alpha', 0.5)
        params.setdefault('edgecolor', 'black')

        conf = context.configuration.to_univariate()

        # Create the bin edges
        bin_edges = conf.uni_histogram_limits()
        # Get the study feature dataframe
        x_values = conf.uni_study_feature_array()

        return DP_Histogram(
            x=x_values,
            bins=bin_edges,
            params=params,
        )
