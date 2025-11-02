"""
DataPlotter class to hold data and give an interface for plotting with matplotlib and plotly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.collections import LineCollection

from faxai.mathing.distribution.Distribution import Distribution
from faxai.utils.coloring import transparent_colormap


class DataPlotter(ABC):
    """
    Abstract Data Plotter class.

    A DataPlotter holds data and provides an interface for plotting with matplotlib and plotly.
    Each DataPlotter instance is responsible for plotting itself on a given axis.
    It should get all the parameters required for plotting during initialization.
    """

    @abstractmethod
    def matplotlib_plot(self, ax: plt.axes) -> None:
        pass

    @abstractmethod
    def plotly_plot(self, ax: go.Figure) -> None:
        pass


class DP_Line(DataPlotter):
    """
    Data Plotter for Lines in 2D
    """

    def __init__(self, x: np.array, y: np.array, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.y = y
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.plot(self.x, self.y, **self.params)

    def plotly_plot(self, fig: go.Figure):
        default = {"mode": "lines"}
        trace_kwargs = {**default, **self.params}
        fig.add_trace(go.Scatter(x=self.x, y=self.y, **trace_kwargs))


class DP_Scatter(DataPlotter):
    """
    Data Plotter for Scatter points in 2D
    """

    def __init__(self, x: np.array, y: np.array, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.y = y
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.scatter(self.x, self.y, **self.params)

    def plotly_plot(self, fig: go.Figure):
        default = {"mode": "markers"}
        trace_kwargs = {**default, **self.params}
        fig.add_trace(go.Scatter(x=self.x, y=self.y, **trace_kwargs))


class DP_Area(DataPlotter):
    """
    Data Plotter for Area Plots
    """

    def __init__(self, x: np.array, y_min: np.array, y_max: np.array, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.y_min = y_min
        self.y_max = y_max
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.fill_between(self.x, self.y_min, self.y_max, **self.params)


class DP_Histogram(DataPlotter):
    """
    Data Plotter for Histogram Plots
    """

    def __init__(self, x: np.array, bins: int = None, params: dict = None, max_height: int = 1, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.params = dict(params) if params else {}
        self.max_height = max_height

        if bins is not None:
            self.params["bins"] = bins

    def matplotlib_plot(self, ax: plt.axes) -> None:
        n, bins, patches = ax.hist(self.x, **self.params)

        # Normalize each patch so that the highest bin becomes 1
        max_height = max(n)
        for patch in patches:
            patch.set_height(patch.get_height() / max_height)


class DP_VerticalLine(DataPlotter):
    """
    Data Plotter for Vertical Line Plots
    """

    def __init__(self, x: float, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.axvline(self.x, **self.params)


class DP_LineCollection(DataPlotter):
    """
    Data Plotter for Collection Plots
    """

    def __init__(self, segments, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.segments = segments
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.add_collection(LineCollection(self.segments, **self.params))


class DP_Collection(DataPlotter):
    """
    Data Plotter for Collection Plots
    """

    def __init__(self, data: List[DataPlotter] = None, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.data = data if data else []
        self.params = dict(params) if params else {}

    def add(self, item: DataPlotter) -> None:
        self.data.append(item)

    def matplotlib_plot(self, ax: plt.axes) -> None:
        for item in self.data:
            item.matplotlib_plot(ax)


class DP_ErrorBar(DataPlotter):
    """
    Data Plotter for Error Bar Plots
    """

    def __init__(self, x: np.array, y: np.array, yerr: np.array, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.x = x
        self.y = y
        self.yerr = yerr
        self.params = dict(params) if params else {}

    def matplotlib_plot(self, ax: plt.axes) -> None:
        ax.errorbar(self.x, self.y, yerr=self.yerr, **self.params)


class DP_NormalDistributionArea(DataPlotter):
    """
    Data Plotter for Distribution Plots
    """

    def __init__(
        self,
        x: np.array,
        mus: np.array,
        stds: np.array,
        max_std: float = 3,
        areas: int = 3,
        params: dict = None,
        axis: int = 0,
    ):
        super().__init__(axis)
        self.x = x
        self.mus = mus
        self.stds = stds
        self.max_std = max_std
        self.areas = areas
        self.params = dict(params) if params else {}

        if "alpha" in self.params:
            self.params["alpha"] /= self.areas
        else:
            self.params["alpha"] = 0.5 / self.areas

        self.plot_areas = []
        self.calculated_ = False

    def calculate(self) -> None:
        self.calculated_ = True

        sigmas = np.linspace(0, self.max_std, self.areas + 1)

        for i, sigma in enumerate(sigmas[1:]):
            y_min = self.mus - sigma * self.stds
            y_max = self.mus + sigma * self.stds

            area = DP_Area(
                x=self.x,
                y_min=y_min,
                y_max=y_max,
                params=self.params,
            )
            self.plot_areas.append(area)

            if "label" in self.params:
                self.params.pop("label")

    def matplotlib_plot(self, ax: plt.axes) -> None:
        if not self.calculated_:
            self.calculate()

        for area in self.plot_areas:
            area.matplotlib_plot(ax)


class DP_VerticalDistribution(DataPlotter):
    """
    Vertical marginal distribution P(x) drawn beside a baseline x = const.
    """

    def __init__(
        self,
        x: float,  # baseline x-position
        distribution: Distribution,  # distribution to use for the curve
        max_width: float = 0.15,  # max horizontal span of the curve (axes units)
        max_current_width: float = 0.15,  # max horizontal span of the curve (axes units)
        side: str = "left",  # 'left' or 'right' of the baseline
        curve_kws: dict | None = None,  # kwargs for the density curve
        line_kws: dict | None = None,  # kwargs for the baseline
        n_points: int = 100,  # number of points for the curve
        axis: int = 0,
    ):
        super().__init__(axis)
        self.x = x
        self.distribution = distribution
        self.max_width = max_width
        self.max_current_width = max_current_width
        self.side = side
        self.n_points = n_points
        self.curve_kws = curve_kws or {}
        self.line_kws = line_kws or {}

    def matplotlib_plot(self, ax: plt.Axes) -> None:
        # Current y-range of the host axis
        y_lo, y_hi = ax.get_ylim()
        y = np.linspace(y_lo, y_hi, self.n_points)

        # Normal pdf and scaling so its max equals `max_width`
        pdf = np.zeros_like(y)
        for i in range(len(y)):
            pdf[i] = self.distribution.pdf(y[i])
        pdf *= self.max_current_width / pdf.max()

        # Place curve on chosen side of baseline
        x_curve = self.x - pdf if self.side == "left" else self.x + pdf
        ax.plot(x_curve, y, **self.curve_kws)

        # Baseline
        ax.axvline(self.x, **self.line_kws)
        if self.side == "left":
            ax.axvline(self.x - self.max_width, **self.line_kws)
        else:
            ax.axvline(self.x + self.max_width, **self.line_kws)


class DP_Distributions(DataPlotter):
    """
    Data Plotter for Distribution Plots
    """

    def __init__(
        self,
        x: np.array,
        distributions: List[Distribution],
        curve_kws: dict | None = None,  # kwargs for the density curve
        line_kws: dict | None = None,  # kwargs for the baseline
        side: str = "left",
        axis: int = 0,
    ):
        super().__init__(axis)
        self.x = x
        self.distributions = distributions
        self.curve_kws = dict(curve_kws) or {}
        self.line_kws = dict(line_kws) or {}
        self.side = side

        self.plot_distributions = []
        self.calculated_ = False

    def calculate(self) -> None:
        self.calculated_ = True

        # Calculate max width as half of the total width between feature values
        max_width = (self.x[0] - self.x[1]) / 2
        max_distribution = max([d.maximum() for d in self.distributions])

        for i, fv in enumerate(self.x):
            max_current_width = max_width * self.distributions[i].maximum() / max_distribution

            dist = DP_VerticalDistribution(
                x=self.x[i],
                distribution=self.distributions[i],
                max_width=max_width,
                max_current_width=max_current_width,
                curve_kws=dict(self.curve_kws),
                line_kws=dict(self.line_kws),
                side=self.side,
            )
            self.plot_distributions.append(dist)

            if "label" in self.curve_kws:
                self.curve_kws.pop("label")

            if "label" in self.line_kws:
                self.line_kws.pop("label")

    def matplotlib_plot(self, ax: plt.axes) -> None:
        if not self.calculated_:
            self.calculate()

        for area in self.plot_distributions:
            area.matplotlib_plot(ax)


class DP_DistributionArea(DataPlotter):
    """
    Data Plotter for Distribution Plots
    """

    def __init__(
        self,
        x: np.array,
        y_limits: tuple,
        distributions: List[Distribution],
        params: dict = None,
        n_points: int = 100,  # number of points for the curve
        axis: int = 0,
    ):
        super().__init__(axis)
        self.x = x
        self.y_limits = y_limits
        self.distributions = distributions
        self.params = dict(params) if params else {}
        self.n_points = n_points

        if "alpha" in self.params:
            self.params["alpha"] = 0.5

        self.calculated_ = False
        self.y = None
        self.matrix = None
        self.dp_background = None

    def calculate(self) -> None:
        self.calculated_ = True

        # Current y-range of the host axis
        self.y = np.linspace(self.y_limits[0], self.y_limits[1], self.n_points)

        self.matrix = np.zeros((len(self.distributions), self.n_points))

        for i, dist in enumerate(self.distributions):
            for j, y in enumerate(self.y):
                self.matrix[i][j] = dist.pdf(y)

        # Normalize the matrix
        max_possible_value = max([d.maximum() for d in self.distributions])

        self.matrix /= max_possible_value
        self.dp_background = DP_Background(x=self.x, y=self.y, matrix=self.matrix, params=self.params)

    def matplotlib_plot(self, ax: plt.axes) -> None:
        if not self.calculated_:
            self.calculate()

        self.dp_background.matplotlib_plot(ax)


class DP_Background(DataPlotter):
    """
    Data Plotter for Background Plots
    """

    def __init__(
        self,
        x: np.array,
        y: np.array,
        matrix: np.ndarray,
        color: str = None,
        params: dict = None,
        axis: int = 0,
    ):
        super().__init__(axis)
        self.x = x
        self.y = y
        self.matrix = matrix
        self.params = dict(params) if params else {}

        if not color:
            if "color" not in self.params:
                self.color = "lightgray"
            else:
                self.color = self.params["color"]
        else:
            self.color = color
        self.params.pop("color")

    def matplotlib_plot(self, ax: plt.axes) -> None:
        cmap = transparent_colormap(self.color)
        extent = (self.x[0], self.x[-1], self.y[0], self.y[-1])

        # Turn the matrix
        matrix = np.transpose(self.matrix)

        # Turn the matrix over horizontal axis
        matrix = np.flip(matrix, axis=0)

        im = ax.imshow(
            matrix,
            # origin="lower",
            extent=extent,
            cmap=cmap,
            **self.params,
        )


class DP_ContinuousLine(DataPlotter):
    """
    Like DP_Line, but treats NaNs as 'carry-forward' flat segments:
      • For each contiguous NaN span, draw a horizontal line at the last observed y.
      • If the NaN span is at the beginning, use the series' last non-NaN value.
      • If all values are NaN, draw y=0 with the normal style.
      • NaN segments use '-.' and half the normal width (Plotly: dashdot, width/2).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, params: dict = None, axis: int = 0):
        super().__init__(axis)

        collection = DP_Collection(params=params, axis=axis)

        # First main line (NaN are ignored by default)
        collection.add(DP_Line(x=x, y=y, params=params, axis=axis))

        # Separate the x and y values into contiguous segments
        y_filled = pd.Series(y).interpolate(limit_direction="both").to_numpy()
        params = params.copy()
        params["linestyle"] = "-."
        if "linewidth" not in params:
            params["linewidth"] = 0.5
        else:
            params["linewidth"] /= 2

        # Remove the label if it exists in params
        params.pop("label", None)

        collection.add(
            DP_Line(
                x=x,
                y=y_filled,
                params=params,
                axis=0,
            )
        )

        self.inner_line = collection

    def matplotlib_plot(self, ax: plt.axes) -> None:
        self.inner_line.matplotlib_plot(ax)

    def plotly_plot(self, fig: go.Figure):
        self.inner_line.plotly_plot(fig)


class DP_WeightedLine(DataPlotter):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        max_width: float = None,
        min_width: float = 0,
        params: dict = None,
        axis: int = 0,
    ):
        super().__init__(axis)

        if max_width is None:
            if "linewidth" in params:
                max_width = params["linewidth"]
            else:
                max_width = 1.0

        params = dict(params) if params else {}

        collection = DP_Collection(params=params, axis=axis)

        # Max and min weight
        max_w = np.max(weights)
        min_w = np.min(weights)

        for i in range(len(x) - 1):
            params = dict(params)
            params["linewidth"] = min_width + (weights[i] - min_w) / (max_w - min_w) * (max_width - min_width)

            collection.add(
                DP_Line(
                    x=np.array([x[i], x[i + 1]]),
                    y=np.array([y[i], y[i + 1]]),
                    params={**params},
                    axis=axis,
                )
            )

            if i == 0:
                params.pop("label", None)

        self.inner_line = collection

    def matplotlib_plot(self, ax: plt.axes) -> None:
        self.inner_line.matplotlib_plot(ax)

    def plotly_plot(self, fig: go.Figure):
        self.inner_line.plotly_plot(fig)
