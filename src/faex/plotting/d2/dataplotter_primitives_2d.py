from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from faex.plotting.plotter_params import LineParams, ScatterParams, HistParams, AreaParams, ErrorBarParams
from faex.plotting.plotter_matplotlib import MatplotlibAxes, mpl_line_kwargs, mpl_scatter_kwargs, mpl_hist_kwargs, mpl_area_kwargs, mpl_errorbar_kwargs
from faex.plotting.plotter_plotly import PlotlyAxes, plotly_line_kwargs, plotly_scatter_kwargs, plotly_hist_kwargs, plotly_area_kwargs, plotly_errorbar_kwargs
from faex.plotting.DataPlotter import DataPlotter

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _as_1d(a: Union[Sequence[float], NDArray[np.floating]]) -> NDArray[np.floating]:
    arr = np.asarray(a)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr


def _check_xy(x: NDArray[np.floating], y: NDArray[np.floating], cls: str) -> None:
    if x.shape != y.shape:
        raise ValueError(f"{cls}: x and y must have the same shape; got {x.shape} vs {y.shape}")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError(f"{cls}: x and y must be 1-D arrays")


# ---------------------------------------------------------------------------
# PRIMITIVE TYPES
# ---------------------------------------------------------------------------

class DP2_Line(DataPlotter):
    """Line plot."""

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating], params: LineParams):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP2_Line")

        # TODO modify this so the arguments are the correct ones
        # If params is a dictionary, convert it to LineParams
        if isinstance(params, dict):
            self.params = LineParams.from_dict(params)
        else:
            self.params = params.copy()


    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        axis.ax.plot(self.x, self.y, **mpl_line_kwargs(self.params))

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        axis.add_trace(go.Scatter(x=self.x, y=self.y, **plotly_line_kwargs(self.params)))


class DP2_Scatter(DataPlotter):
    """Scatter plot."""

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating], params: ScatterParams):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP2_Scatter")

        # TODO modify this so the arguments are the correct ones
        # If params is a dictionary, convert it to LineParams
        if isinstance(params, dict):
            self.params = ScatterParams.from_dict(params)
        else:
            self.params = params.copy()

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        axis.ax.scatter(self.x, self.y, **mpl_scatter_kwargs(self.params))

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        axis.add_trace(go.Scatter(x=self.x, y=self.y, **plotly_scatter_kwargs(self.params)))


class DP2_Area(DataPlotter):
    """Filled area between `y_min` and `y_max` over `x`."""

    def __init__(
        self,
        x: NDArray[np.floating],
        y_min: NDArray[np.floating],
        y_max: NDArray[np.floating],
        params: AreaParams,
    ):
        self.x = _as_1d(x)
        self.y_min = _as_1d(y_min)
        self.y_max = _as_1d(y_max)
        _check_xy(self.x, self.y_min, "DP2_Area")
        _check_xy(self.x, self.y_max, "DP2_Area")

        # TODO modify this so the arguments are the correct ones
        # If params is a dictionary, convert it to LineParams
        if isinstance(params, dict):
            self.params = AreaParams.from_dict(params)
        else:
            self.params = params.copy()


    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        axis.ax.fill_between(self.x, self.y_min, self.y_max, **mpl_area_kwargs(self.params))

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        kwargs = plotly_area_kwargs(self.params)

        axis.add_trace(go.Scatter(x=self.x, y=self.y_min, fill="none", **kwargs))

        axis.add_trace(go.Scatter(x=self.x, y=self.y_max, fill="tonexty", **kwargs))


class DP2_Histogram(DataPlotter):
    """
    Histogram with optional max-height normalization (bar heights scaled so max bin == `max_height`).
    Note: This is different from `density=True` which normalizes area under the histogram.
    """

    def __init__(
        self,
        x: NDArray[np.floating],
        bins: Optional[Union[int, Sequence[float]]] = None,
        params: Optional[Dict[str, Any]] = None,
        max_height: float = 1.0,
    ):
        self.x = _as_1d(x)
        self.bins = bins
        self.max_height = float(max_height)

        if isinstance(params, dict):
            self.params = HistParams.from_dict(params)
        else:
            self.params = params.copy()

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        # Draw normally, then scale bars so that max == self.max_height
        if self.bins is not None:
            self.params.bins = self.bins

        counts, bin_edges, patches = axis.ax.hist(self.x, **mpl_hist_kwargs(self.params))
        if counts.size == 0:
            return

        peak = float(np.max(counts))
        if peak <= 0.0:
            return

        scale = self.max_height / peak
        for rect, c in zip(patches, counts):
            rect.set_height(c * scale)
            rect.set_y(0.0)

        # Ensure y-limit is high enough to show the scaled bars
        # ymin, ymax = axis.ax.get_ylim()
        # if ymax < self.max_height:
        #     axis.ax.set_ylim(ymin, max(self.max_height, ymax))


    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        # Compute histogram ourselves to support "max-height" normalization
        if self.bins is None:
            bins = "auto"
        else:
            bins = self.bins

        # Resolve bins -> numeric edges
        if isinstance(bins, str):
            # Handle a couple of typical strategies
            edges = np.histogram_bin_edges(self.x, bins=bins)
        elif isinstance(bins, int):
            edges = np.histogram_bin_edges(self.x, bins=bins)
        else:
            edges = np.asarray(bins, dtype=float)

        counts, edges = np.histogram(self.x, bins=edges)
        if counts.size == 0:
            return

        peak = float(np.max(counts))
        if peak <= 0.0:
            return

        scale = self.max_height / peak
        heights = counts.astype(float) * scale
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = edges[1:] - edges[:-1]

        style = plotly_hist_kwargs(self.params)
        # Map Matplotlib alpha -> Plotly marker opacity if not already mapped
        marker = style.pop("marker", {})
        line = style.pop("line", {})
        # Bars: color maps through marker.color; outline through marker.line
        if "color" in line and "color" not in marker:
            marker["color"] = line["color"]
        marker.setdefault("line", {})
        if "width" in line:
            marker["line"]["width"] = line["width"]
        if "dash" in line:
            # Bars don't support dash; ignore
            pass

        axis.add_trace(go.Bar(x=centers, y=heights, width=widths, marker=marker, **style))


class DP2_ErrorBar(DataPlotter):
    """Error bar plot (typically scatter with error bars)."""

    def __init__(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        error: NDArray[np.floating],
        params: ErrorBarParams,
    ):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        self.error = _as_1d(error)
        _check_xy(self.x, self.y, "DP2_ErrorBar")
        _check_xy(self.x, self.error, "DP2_ErrorBar")

        # If params is a dictionary, convert it to ErrorBarParams
        if isinstance(params, dict):
            self.params = ErrorBarParams.from_dict(params)
        else:
            self.params = params.copy()

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        # expects something like mpl_errorbar_kwargs(ErrorBarParams)
        axis.ax.errorbar(self.x, self.y, yerr=self.error, **mpl_errorbar_kwargs(self.params))

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        # expects something like plotly_errorbar_kwargs(ErrorBarParams)
        axis.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                error_y={"type": "data", "array": self.error},
                **plotly_errorbar_kwargs(self.params),
            )
        )
