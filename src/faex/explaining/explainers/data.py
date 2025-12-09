"""
Submodule to handle explainers related to the actual data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np

from faex.plotting.d2.dataplotter_primitives_2d import DP2_Histogram, DP2_Scatter
from faex.explaining.Explainer import ExplainerPlot
from faex.explaining.ExplainerFactory import ExplainerFactory

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class RealPrediction(ExplainerPlot):
    # TODO check study features set and that is 1d

    def plot(self, context: ExplainerContext, params: dict = None) -> DP2_Scatter:
        params = dict(params) if params else {}

        params.setdefault("color", "navy")
        params.setdefault("opacity", 0.5)
        params.setdefault("label", "predicted")
        params.setdefault("size", 5)

        # Get the study feature dataframe
        df = context.configuration.study_feature_dataframe()

        # Check it is only one column
        if df.shape[1] != 1:
            raise ValueError("RealPrediction plot requires a single study feature.")
        x_values = df.iloc[:, 0]

        return DP2_Scatter(x=x_values, y=context.configuration.get_real_predictions(), params=params)


# Register Explainer
ExplainerFactory.register_explainer(
    explainer=RealPrediction,
    aliases=["real", "real-prediction", "real-predictions"],
)


class Histogram(ExplainerPlot):

    def plot(self, context: ExplainerContext, params: dict = None) -> DP2_Scatter:
        params = dict(params) if params else {}

        params.setdefault("density", True)
        params.setdefault("color", "aquamarine")
        params.setdefault("opacity", 0.5)
        params.setdefault("edgecolor", "black")

        conf = context.configuration.to_univariate()

        # Create the bin edges
        bin_edges = conf.uni_histogram_limits()
        # Get the study feature dataframe
        x_values = conf.uni_study_feature_array()

        params.setdefault("bins", bin_edges)

        return DP2_Histogram(
            x=x_values,
            params=params,
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=Histogram,
    aliases=["histogram", "hist"],
)


class Distribution(ExplainerPlot):
    # TODO check study features set and that is 1d

    def plot(self, context: ExplainerContext, params: dict = None) -> DP2_Scatter:

        params = dict(params) if params else {}

        params.setdefault('color', 'black')
        params.setdefault('alpha', 0.7)
        params.setdefault('size', 15)
        params.setdefault('marker', '|')

        # Get the study feature dataframe
        df = context.configuration.study_feature_dataframe()

        # Check it is only one column
        if df.shape[1] != 1:
            raise ValueError("RealPrediction plot requires a single study feature.")
        x_values = df.iloc[:, 0]

        return DP2_Scatter(
            x=x_values,
            y=np.ones(len(x_values)) * 0.1,
            params=params,
        )

# Register Explainer
ExplainerFactory.register_explainer(
    explainer=Distribution,
    aliases=["values"],
)
