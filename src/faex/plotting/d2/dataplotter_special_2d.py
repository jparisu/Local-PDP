from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.collections import LineCollection as MplLineCollection
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from faex.plotting.plotter_params import LineParams
from faex.plotting.d2.dataplotter_primitives_2d import DP2_Line
from faex.plotting.plotter_matplotlib import MatplotlibAxes, mpl_line_kwargs, mpl_scatter_kwargs, mpl_hist_kwargs, mpl_area_kwargs
from faex.plotting.plotter_plotly import PlotlyAxes, plotly_line_kwargs, plotly_scatter_kwargs, plotly_hist_kwargs, plotly_area_kwargs
from faex.plotting.DataPlotter import DataPlotter, DP2_Collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_LineStyleToDash = {"-": None, "--": "dash", "-.": "dashdot", ":": "dot"}

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
# SPECIAL CLASSES
# ---------------------------------------------------------------------------


class DP2_WeightedWidthLine(DataPlotter):
    """
    Piecewise line where the width of each segment is scaled by `weights[i]`.

    Notes
    -----
    - Expects len(weights) == len(x) - 1 (one weight per segment).
    - If all weights are equal, draws all segments with `max_width`.
    """

    def __init__(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        weights: NDArray[np.floating],
        params: LineParams,
        max_width: Optional[float] = None,
        min_width: float = 0.0,
    ):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        self.weights = _as_1d(weights)
        _check_xy(self.x, self.y, "DP2_WeightedWidthLine")
        _check_xy(self.x, self.weights, "DP2_WeightedWidthLine")

        # TODO modify this so the arguments are the correct ones
        # If params is a dictionary, convert it to LineParams
        if isinstance(params, dict):
            params = LineParams.from_dict(params)

        if max_width is None:
            max_width = params.linewidth
        if max_width is None:
            max_width = 1.0

        self.min_width = float(min_width)
        self.max_width = float(max_width)

        # Build a collection of small DP2_Line segments
        collection = DP2_Collection()
        w_max = float(np.max(self.weights))
        w_min = float(np.min(self.weights))
        denom = w_max - w_min

        for i in range(self.x.size - 1):
            # Compute width safely
            if denom <= 0.0:
                width_i = self.max_width
            else:
                width_i = self.min_width + (self.weights[i] - w_min) / denom * (self.max_width - self.min_width)

            params_i = params.copy()
            params_i.linewidth = width_i
            if i > 0:
                params_i.label = None

            collection.add(
                DP2_Line(
                    x=np.array([self.x[i], self.x[i + 1]], dtype=float),
                    y=np.array([self.y[i], self.y[i + 1]], dtype=float),
                    params=params_i,
                )
            )

        self._inner = collection

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        self._inner._matplotlib_plot(axis)

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        self._inner._plotly_plot(axis)


class DP2_WeightedOpacityLine(DataPlotter):
    """
    Piecewise line where the width of each segment is scaled by `weights[i]`.

    Notes
    -----
    - Expects len(weights) == len(x) - 1 (one weight per segment).
    - If all weights are equal, draws all segments with `max_width`.
    """

    def __init__(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        weights: NDArray[np.floating],
        params: LineParams,
        max_opacity: float = None,
        min_opacity: float = 0.0,
    ):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        self.weights = _as_1d(weights)
        _check_xy(self.x, self.y, "DP2_WeightedWidthLine")
        _check_xy(self.x, self.weights, "DP2_WeightedWidthLine")

        # TODO modify this so the arguments are the correct ones
        # If params is a dictionary, convert it to LineParams
        if isinstance(params, dict):
            params = LineParams.from_dict(params)

        if max_opacity is None:
            max_opacity = params.opacity
        if max_opacity is None:
            max_opacity = 1.0

        self.min_opacity = float(min_opacity)
        self.max_opacity = float(max_opacity)

        # Build a collection of small DP2_Line segments
        collection = DP2_Collection()
        w_max = float(np.max(self.weights))
        w_min = float(np.min(self.weights))
        denom = w_max - w_min

        for i in range(self.x.size - 1):
            # Compute width safely
            if denom <= 0.0:
                width_i = self.max_opacity
            else:
                width_i = self.min_opacity + (self.weights[i] - w_min) / denom * (self.max_opacity - self.min_opacity)

            params_i = params.copy()
            params_i.opacity = width_i
            if i > 0:
                params_i.label = None

            collection.add(
                DP2_Line(
                    x=np.array([self.x[i], self.x[i + 1]], dtype=float),
                    y=np.array([self.y[i], self.y[i + 1]], dtype=float),
                    params=params_i,
                )
            )

        self._inner = collection

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        self._inner._matplotlib_plot(axis)

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        self._inner._plotly_plot(axis)



# class DP2_LineCollection(DataPlotter):
#     """
#     A collection of disjoint line segments.

#     `segments` may be:
#       - an array-like of shape (N, 2, 2): [[[x0, y0], [x1, y1]], ...]
#       - a list of 2-tuples: [((x0, y0), (x1, y1)), ...]
#     """

#     def __init__(self, segments: Iterable[Iterable[Iterable[float]]], params: Optional[Dict[str, Any]] = None):
#         self.segments = list(segments)
#         self.params = params.copy()

#     def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
#         axis.ax.add_collection(MplLineCollection(self.segments, **mpl_line_kwargs(self.params)))

#     def _plotly_plot(self, axis: PlotlyAxes) -> None:
#         # Efficiently plot multiple segments in one trace by separating with Nones
#         xs: List[float] = []
#         ys: List[float] = []
#         for seg in self.segments:
#             (x0, y0), (x1, y1) = seg
#             xs.extend([x0, x1, None])
#             ys.extend([y0, y1, None])

#         kwargs = {"mode": "lines", **plotly_line_kwargs(self.params)}
#         axis.fig.add_trace(go.Scatter(x=xs, y=ys, **kwargs))



# class DP2_VerticalLine(DataPlotter):
#     """Single vertical line at `x`."""

#     def __init__(self, x: float, params: LineParams):
#         self.x = float(x)
#         self.params = params.copy()

#     def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
#         axis.ax.axvline(self.x, **mpl_line_kwargs(self.params))

#     def _plotly_plot(self, axis: PlotlyAxes) -> None:
#         # Get arguments for line style

#         axis.fig.add_vline(
#             x=self.x,
#             line_color=self.params.color,
#             line_width=self.params.linewidth,
#             line_dash=self.params.style,
#             opacity=self.params.opacity,
#             row=axis.row,
#             col=axis.col,
#         )




# class DP2_ErrorBar(DataPlotter):
#     """Errorbar plot where `yerr` are symmetric absolute errors."""

#     def __init__(
#         self,
#         x: NDArray[np.floating],
#         y: NDArray[np.floating],
#         yerr: NDArray[np.floating],
#         params: Optional[Dict[str, Any]] = None,
#     ):
#         self.x = _as_1d(x)
#         self.y = _as_1d(y)
#         self.yerr = _as_1d(yerr)
#         _check_xy(self.x, self.y, "DP2_ErrorBar")
#         if self.yerr.shape != self.y.shape:
#             raise ValueError("DP2_ErrorBar: yerr must match y shape")
#         self.params = params.copy()

#     def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
#         ax.ax.errorbar(self.x, self.y, yerr=self.yerr, **mpl_line_kwargs(self.params))

#     def _plotly_plot(self, axis: PlotlyAxes) -> None:
#         style = _mpl_to_plotly_style(self.params)
#         # Default to markers unless overridden by user
#         mode = style.pop("mode", "markers")
#         fig.add_trace(
#             go.Scatter(
#                 x=self.x,
#                 y=self.y,
#                 mode=mode,
#                 error_y=dict(type="data", array=self.yerr, visible=True),
#                 **style,
#             )
#         )


# class DP2_NormalDistributionArea(DataPlotter):
#     """
#     Shades ±k·σ bands around a time-varying mean μ(x).
#     For `areas=3` and `max_std=3`, this draws 1σ, 2σ, and 3σ bands.

#     Parameters
#     ----------
#     x : array-like of shape (n,)
#         Domain along which μ and σ are defined.
#     mus : array-like of shape (n,)
#         Mean at each x.
#     stds : array-like of shape (n,)
#         Standard deviation at each x.
#     max_std : float, default 3
#         Largest sigma band to draw.
#     areas : int, default 3
#         Number of sigma bands between 0 and `max_std`.
#     """

#     def __init__(
#         self,
#         x: NDArray[np.floating],
#         mus: NDArray[np.floating],
#         stds: NDArray[np.floating],
#         max_std: float = 3.0,
#         areas: int = 3,
#         params: Optional[Dict[str, Any]] = None,
#     ):
#         self.x = _as_1d(x)
#         self.mus = _as_1d(mus)
#         self.stds = _as_1d(stds)
#         _check_xy(self.x, self.mus, "DP2_NormalDistributionArea")
#         _check_xy(self.x, self.stds, "DP2_NormalDistributionArea")

#         self.max_std = float(max_std)
#         self.areas = int(areas)
#         base_params = _copy_params(params)

#         # Split transparency across bands (do not mutate caller params)
#         alpha = base_params.get("opacity", 0.5)
#         self.band_params: Dict[str, Any] = dict(base_params)
#         self.band_params["opacity"] = alpha / max(self.areas, 1)

#         self.plot_areas: List[DP2_Area] = []
#         self._calculated = False

#     def calculate(self) -> None:
#         self._calculated = True
#         self.plot_areas.clear()

#         sigmas = np.linspace(0.0, self.max_std, self.areas + 1)  # [0, 1σ, 2σ, ...]
#         band_params = dict(self.band_params)

#         # Do not repeat label across multiple bands
#         band_params.pop("label", None)

#         for sigma in sigmas[1:]:
#             y_min = self.mus - sigma * self.stds
#             y_max = self.mus + sigma * self.stds
#             self.plot_areas.append(DP2_Area(x=self.x, y_min=y_min, y_max=y_max, params=band_params))

#     def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
#         if not self._calculated:
#             self.calculate()
#         for area in self.plot_areas:
#             area._matplotlib_plot(ax)

#     def _plotly_plot(self, axis: PlotlyAxes) -> None:
#         if not self._calculated:
#             self.calculate()
#         for area in self.plot_areas:
#             area._plotly_plot(fig)


# class DP2_ContinuousLine(DataPlotter):
#     """
#     Like DP2_Line, but treats NaNs as 'carry-forward' flat segments:

#       • For each contiguous NaN span, draw a horizontal line at the last observed y.
#       • If the NaN span is at the beginning, use the series' last non-NaN value.
#       • If all values are NaN, draw y=0 with the normal style.
#       • NaN segments use '-.' and half the normal width (Plotly dashdot, width/2).
#     """

#     def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating], params: Optional[Dict[str, Any]] = None):
#         self.x = _as_1d(x)
#         self.y = _as_1d(y)
#         _check_xy(self.x, self.y, "DP2_ContinuousLine")

#         base_params = _copy_params(params)

#         collection = DP2_Collection()
#         # Main line (NaNs break segments naturally in both backends)
#         collection.add(DP2_Line(self.x, self.y, params=base_params))

#         # Dashed segments only over NaN spans (mask non-NaN as gaps)
#         y_ffill = _forward_fill_for_continuous(self.y)
#         dashed_y = np.where(np.isnan(self.y), y_ffill, np.nan)

#         dashed_params = dict(base_params)
#         # style: '-.' and half width
#         dashed_params["linestyle"] = "-."
#         if "linewidth" in dashed_params:
#             dashed_params["linewidth"] = dashed_params["linewidth"] / 2.0
#         else:
#             dashed_params["linewidth"] = 0.5
#         dashed_params.pop("label", None)

#         collection.add(DP2_Line(self.x, dashed_y, params=dashed_params))

#         self._inner = collection

#     def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
#         self._inner._matplotlib_plot(ax)

#     def _plotly_plot(self, axis: PlotlyAxes) -> None:
#         self._inner._plotly_plot(fig)
