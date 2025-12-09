"""
Module for the class handler to hold explanations and configurations easy for the user.
"""

from __future__ import annotations

import logging
from typing import Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from faex.core.DataCore import DataCore, UnivariateDataCore
from faex.plotting.DataPlotter import PlotlyAxes
from faex.explaining.ExplainerContext import ExplainerContext
from faex.explaining.ExplainerFactory import ExplainerFactory, GlobalExplainerFactory
from faex.plotting.plotter_matplotlib import MatplotlibAxes, generate_default_double_figure_matplotlib

logger = logging.getLogger(__name__)

class ExplanationCore:

    def __init__(self, datacore: DataCore):
        """
        Initializes the ExplanationCore with an explanation and its configuration.

        Args:
            datacore (DataCore): The data core containing the data to be explained.
        """

        # Check datacore is univariate
        if len(datacore.study_features) != 1:
            NotImplementedError("Only univariate data cores are supported in ExplanationCore.")

        self._unidatacore = datacore.to_univariate()
        self._context = ExplainerContext(self._unidatacore)


    def visualize_doubleplot(
            self,
            explanations: list[str],
            plot_arguments: dict[str, any] | None = None,
            matplotlib: bool = True,
    ) -> Any:
        if matplotlib:
            return self.visualize_doubleplot_matplotlib(
                explanations=explanations,
                plot_arguments=plot_arguments,
            )
        else:
            return self.visualize_doubleplot_plotly(
                explanations=explanations,
                plot_arguments=plot_arguments,
            )


    def visualize_doubleplot_matplotlib(
            self,
            explanations: list[str],
            plot_arguments: dict[str, any] | None = None,
    ) -> plt.Figure:

        x_limits: Tuple[float, float] = self._unidatacore.uni_feature_limits()

        ice = self._context.explain("ice")

        y_limits: Tuple[float, float] = (ice.min(), ice.max())
        x_label: str = self._unidatacore.uni_study_feature()
        y_label: str = "Predictions"
        y2_label: str = "Distributions"

        logger.debug(f"setting limits  x: {x_limits}  y: {y_limits}")

        axis_top, axis_bottom = generate_default_double_figure_matplotlib(
            x_limits=x_limits,
            y_limits=y_limits,
            x_label=x_label,
            y_label=y_label,
            y2_label=y2_label,
        )

        for explanation in explanations:
            args = plot_arguments.get(explanation, {}) if plot_arguments else {}

            plotter = self._context.plot(explanation, params=args)

            if self.__goes_bottom_plot(explanation):
                plotter.matplotlib_plot(axis_bottom)

            else:
                plotter.matplotlib_plot(axis_top)

        axis_top.ax.legend(loc='upper left')

        return axis_top.fig


    def visualize_doubleplot_plotly(
            self,
            explanations: List[str],
            plot_arguments: Dict[str, Any] | None = None,
    ) -> go.Figure:
        """
        Plot the same double-plot layout as `visualize_doubleplot`, but using Plotly.

        Returns
        -------
        fig : go.Figure
            Plotly figure with two rows:
            - row 1: main plot (ICE, predictions, etc.)
            - row 2: bottom plot (e.g. distributions) for some explanations
        """

        # Get limits and labels from your existing helpers/context
        x_limits = self._unidatacore.uni_feature_limits()  # (min_x, max_x)

        ice = self._context.explain("ice")
        y_limits = (float(ice.min()), float(ice.max()))

        x_label: str = self._unidatacore.uni_study_feature()
        y_label: str = "Predictions"
        y2_label: str = "Distributions"

        # Create a 2-row subplot with shared x-axis
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.06,
            row_heights=[0.8, 0.2],
        )

        # Axes ranges and labels
        fig.update_xaxes(
            range=list(x_limits),
            title_text=x_label,
            row=2,
            col=1,
        )

        fig.update_yaxes(
            range=list(y_limits),
            title_text=y_label,
            row=1,
            col=1,
        )

        fig.update_yaxes(
            title_text=y2_label,
            row=2,
            col=1,
        )

        # Add traces for each explanation using your existing context/plotter
        for explanation in explanations:
            args = plot_arguments.get(explanation, {}) if plot_arguments else {}

            plotter = self._context.plot(explanation, params=args)

            # Bottom plot (e.g. distributions) if applicable
            if self.__goes_bottom_plot(explanation):
                # Assumes your plotter knows how to add Plotly traces to a given row/col
                ax = PlotlyAxes(fig=fig, row=2, col=1)
            else:
                ax = PlotlyAxes(fig=fig, row=1, col=1)

            plotter.plotly_plot(ax)

        # Layout tweaks (optional)
        fig.update_layout(
            margin=dict(l=60, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            # template="plotly_white",
            # paper_bgcolor='white', plot_bgcolor='white'
        )

        return fig

    @staticmethod
    def __goes_bottom_plot(name: str):
        return GlobalExplainerFactory.get_instance().name_convention(name) in ExplanationCore.__bottom_plot()


    @staticmethod
    def __bottom_plot():
        return [
            "histogram",
            "kernelnormalizer",
            "distribution",
        ]
