from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection as MplLineCollection

import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def _copy_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(params) if params else {}


def _mpl_to_plotly_style(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate a subset of common Matplotlib-style kwargs to Plotly kwargs.
    - alpha -> opacity
    - color -> line.color & marker.color (if relevant)
    - linewidth/lw -> line.width
    - linestyle/ls -> line.dash
    """
    out: Dict[str, Any] = {}
    p = dict(params)

    label = p.pop("label", None)

    alpha = p.pop("alpha", None)
    if alpha is not None:
        out["opacity"] = alpha

    color = p.pop("color", p.pop("c", None))
    linewidth = p.pop("linewidth", p.pop("lw", None))
    linestyle = p.pop("linestyle", p.pop("ls", None))

    # Line dict
    line: Dict[str, Any] = {}
    if linewidth is not None:
        line["width"] = linewidth
    if linestyle is not None:
        dash = _LineStyleToDash.get(linestyle)
        if dash:
            line["dash"] = dash
    if color is not None:
        line["color"] = color
    if line:
        out["line"] = line

    # Some users pass marker='o' etc.; set symbol if present
    marker = p.pop("marker", None)
    if marker is not None:
        out.setdefault("marker", {})
        out["marker"]["symbol"] = marker
        if color is not None:
            out["marker"]["color"] = color

    # Keep any remaining kwargs that Plotly will understand (e.g., hovertemplate)
    out.update(p)
    return out


def _forward_fill_for_continuous(y: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Carry-forward fill for NaN segments:
    - Fill each NaN with the previous non-NaN value.
    - If leading NaNs exist, use the last non-NaN value in the series.
    - If all values are NaN, return zeros.
    """
    y = np.asarray(y, dtype=float).copy()
    n = y.size
    if n == 0:
        return y

    mask = np.isnan(y)
    if mask.all():
        return np.zeros_like(y)

    # Seed the first value if it is NaN: use last non-NaN in the series
    if np.isnan(y[0]):
        last_idx = np.where(~mask)[0][-1]
        y[0] = y[last_idx]

    for i in range(1, n):
        if np.isnan(y[i]):
            y[i] = y[i - 1]
    return y


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DataPlotter(ABC):
    """
    Base interface for lightweight plotting adapters that can render to
    both Matplotlib and Plotly.
    """

    def matplotlib_plot(self, ax: plt.Axes | None = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render this plotter onto a Matplotlib Axes.
        If `ax` is None, creates a new figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots()
        self._matplotlib_plot(ax)
        return fig, ax

    def plotly_plot(self, fig: go.Figure | None = None) -> go.Figure:
        """
        Render this plotter onto a Plotly Figure.
        If `fig` is None, creates a new figure.
        """
        if fig is None:
            fig = make_subplots()
        self._plotly_plot(fig)
        return fig

    @abstractmethod
    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        """Render this plotter onto a Matplotlib Axes."""

    @abstractmethod
    def _plotly_plot(self, fig: go.Figure) -> None:
        """Render this plotter onto a Plotly Figure (optionally to a given subplot)."""


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class DP_Line(DataPlotter):
    """Line plot."""

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating],
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP_Line")
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.plot(self.x, self.y, **self.params)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        kwargs = {"mode": "lines", **_mpl_to_plotly_style(self.params)}
        fig.add_trace(go.Scatter(x=self.x, y=self.y, **kwargs), row=row, col=col)


class DP_Scatter(DataPlotter):
    """Scatter plot."""

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating],
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP_Scatter")
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.scatter(self.x, self.y, **self.params)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        kwargs = {"mode": "markers", **_mpl_to_plotly_style(self.params)}
        fig.add_trace(go.Scatter(x=self.x, y=self.y, **kwargs), row=row, col=col)


class DP_Area(DataPlotter):
    """Filled area between `y_min` and `y_max` over `x`."""

    def __init__(self, x: NDArray[np.floating], y_min: NDArray[np.floating], y_max: NDArray[np.floating],
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y_min = _as_1d(y_min)
        self.y_max = _as_1d(y_max)
        _check_xy(self.x, self.y_min, "DP_Area")
        _check_xy(self.x, self.y_max, "DP_Area")
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.fill_between(self.x, self.y_min, self.y_max, **self.params)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        # Build a closed polygon (x forward + x reversed)
        x_poly = np.concatenate([self.x, self.x[::-1]])
        y_poly = np.concatenate([self.y_min, self.y_max[::-1]])
        kwargs = _mpl_to_plotly_style(self.params)
        kwargs["mode"] = kwargs.get("mode", "lines")
        kwargs["fill"] = "toself"
        fig.add_trace(go.Scatter(x=x_poly, y=y_poly, **kwargs), row=row, col=col)


class DP_Histogram(DataPlotter):
    """
    Histogram with optional max-height normalization (bar heights scaled so max bin == `max_height`).
    Note: This is different from `density=True` which normalizes area under the histogram.
    """

    def __init__(self, x: NDArray[np.floating], bins: Optional[Union[int, Sequence[float]]] = None,
                 params: Optional[Dict[str, Any]] = None, max_height: float = 1.0,
                 ):
        self.x = _as_1d(x)
        self.bins = bins
        self.params = _copy_params(params)
        self.max_height = float(max_height)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        # Draw normally, then scale bars so that max == self.max_height
        hist_kwargs = dict(self.params)
        if self.bins is not None:
            hist_kwargs["bins"] = self.bins

        counts, bin_edges, patches = ax.hist(self.x, **hist_kwargs)
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
        ymin, ymax = ax.get_ylim()
        if ymax < self.max_height:
            ax.set_ylim(ymin, max(self.max_height, ymax))

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
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
        widths = (edges[1:] - edges[:-1])

        style = _mpl_to_plotly_style(self.params)
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

        fig.add_trace(
            go.Bar(x=centers, y=heights, width=widths, marker=marker, **style),
            row=row, col=col
        )


class DP_VerticalLine(DataPlotter):
    """Single vertical line at `x`."""

    def __init__(self, x: float, params: Optional[Dict[str, Any]] = None):
        self.x = float(x)
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.axvline(self.x, **self.params)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        # Plotly shapes (add_vline) don't appear in legend; this mirrors Matplotlib axvline visuals.
        line_dash = _LineStyleToDash.get(self.params.get("linestyle", None), None)
        fig.add_vline(
            x=self.x,
            line_color=self.params.get("color", None),
            line_width=self.params.get("linewidth", self.params.get("lw", None)),
            line_dash=line_dash,
            opacity=self.params.get("alpha", None),
            row=row, col=col
        )


class DP_LineCollection(DataPlotter):
    """
    A collection of disjoint line segments.

    `segments` may be:
      - an array-like of shape (N, 2, 2): [[[x0, y0], [x1, y1]], ...]
      - a list of 2-tuples: [((x0, y0), (x1, y1)), ...]
    """

    def __init__(self, segments: Iterable[Iterable[Iterable[float]]],
                 params: Optional[Dict[str, Any]] = None):
        self.segments = list(segments)
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.add_collection(MplLineCollection(self.segments, **self.params))

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        # Efficiently plot multiple segments in one trace by separating with Nones
        xs: List[float] = []
        ys: List[float] = []
        for seg in self.segments:
            (x0, y0), (x1, y1) = seg
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])

        kwargs = {"mode": "lines", **_mpl_to_plotly_style(self.params)}
        fig.add_trace(go.Scatter(x=xs, y=ys, **kwargs), row=row, col=col)


class DP_Collection(DataPlotter):
    """A container for other plotters."""

    def __init__(self, data: Optional[List[DataPlotter]] = None,
                 params: Optional[Dict[str, Any]] = None):
        self.data: List[DataPlotter] = list(data) if data else []
        self.params = _copy_params(params)

    def add(self, item: DataPlotter) -> None:
        self.data.append(item)

    def extend(self, items: Iterable[DataPlotter]) -> None:
        self.data.extend(items)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        for item in self.data:
            item._matplotlib_plot(ax)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        for item in self.data:
            item._plotly_plot(fig, row=row, col=col)


class DP_ErrorBar(DataPlotter):
    """Errorbar plot where `yerr` are symmetric absolute errors."""

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating], yerr: NDArray[np.floating],
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        self.yerr = _as_1d(yerr)
        _check_xy(self.x, self.y, "DP_ErrorBar")
        if self.yerr.shape != self.y.shape:
            raise ValueError("DP_ErrorBar: yerr must match y shape")
        self.params = _copy_params(params)

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        ax.errorbar(self.x, self.y, yerr=self.yerr, **self.params)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        style = _mpl_to_plotly_style(self.params)
        # Default to markers unless overridden by user
        mode = style.pop("mode", "markers")
        fig.add_trace(
            go.Scatter(
                x=self.x,
                y=self.y,
                mode=mode,
                error_y=dict(type="data", array=self.yerr, visible=True),
                **style,
            ),
            row=row, col=col
        )


class DP_NormalDistributionArea(DataPlotter):
    """
    Shades ±k·σ bands around a time-varying mean μ(x).
    For `areas=3` and `max_std=3`, this draws 1σ, 2σ, and 3σ bands.

    Parameters
    ----------
    x : array-like of shape (n,)
        Domain along which μ and σ are defined.
    mus : array-like of shape (n,)
        Mean at each x.
    stds : array-like of shape (n,)
        Standard deviation at each x.
    max_std : float, default 3
        Largest sigma band to draw.
    areas : int, default 3
        Number of sigma bands between 0 and `max_std`.
    """

    def __init__(self, x: NDArray[np.floating], mus: NDArray[np.floating], stds: NDArray[np.floating],
                 max_std: float = 3.0, areas: int = 3,
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.mus = _as_1d(mus)
        self.stds = _as_1d(stds)
        _check_xy(self.x, self.mus, "DP_NormalDistributionArea")
        _check_xy(self.x, self.stds, "DP_NormalDistributionArea")

        self.max_std = float(max_std)
        self.areas = int(areas)
        base_params = _copy_params(params)

        # Split transparency across bands (do not mutate caller params)
        alpha = base_params.get("alpha", 0.5)
        self.band_params: Dict[str, Any] = dict(base_params)
        self.band_params["alpha"] = alpha / max(self.areas, 1)

        self.plot_areas: List[DP_Area] = []
        self._calculated = False

    def calculate(self) -> None:
        self._calculated = True
        self.plot_areas.clear()

        sigmas = np.linspace(0.0, self.max_std, self.areas + 1)  # [0, 1σ, 2σ, ...]
        band_params = dict(self.band_params)

        # Do not repeat label across multiple bands
        band_params.pop("label", None)

        for sigma in sigmas[1:]:
            y_min = self.mus - sigma * self.stds
            y_max = self.mus + sigma * self.stds
            self.plot_areas.append(
                DP_Area(x=self.x, y_min=y_min, y_max=y_max, params=band_params)
            )

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        if not self._calculated:
            self.calculate()
        for area in self.plot_areas:
            area._matplotlib_plot(ax)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        if not self._calculated:
            self.calculate()
        for area in self.plot_areas:
            area._plotly_plot(fig, row=row, col=col)


class DP_ContinuousLine(DataPlotter):
    """
    Like DP_Line, but treats NaNs as 'carry-forward' flat segments:

      • For each contiguous NaN span, draw a horizontal line at the last observed y.
      • If the NaN span is at the beginning, use the series' last non-NaN value.
      • If all values are NaN, draw y=0 with the normal style.
      • NaN segments use '-.' and half the normal width (Plotly dashdot, width/2).
    """

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating],
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP_ContinuousLine")

        base_params = _copy_params(params)

        collection = DP_Collection()
        # Main line (NaNs break segments naturally in both backends)
        collection.add(DP_Line(self.x, self.y, params=base_params))

        # Dashed segments only over NaN spans (mask non-NaN as gaps)
        y_ffill = _forward_fill_for_continuous(self.y)
        dashed_y = np.where(np.isnan(self.y), y_ffill, np.nan)

        dashed_params = dict(base_params)
        # style: '-.' and half width
        dashed_params["linestyle"] = "-."
        if "linewidth" in dashed_params:
            dashed_params["linewidth"] = dashed_params["linewidth"] / 2.0
        else:
            dashed_params["linewidth"] = 0.5
        dashed_params.pop("label", None)

        collection.add(DP_Line(self.x, dashed_y, params=dashed_params))

        self._inner = collection

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        self._inner._matplotlib_plot(ax)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        self._inner._plotly_plot(fig, row=row, col=col)


class DP_WeightedLine(DataPlotter):
    """
    Piecewise line where the width of each segment is scaled by `weights[i]`.

    Notes
    -----
    - Expects len(weights) == len(x) - 1 (one weight per segment).
    - If all weights are equal, draws all segments with `max_width`.
    """

    def __init__(self, x: NDArray[np.floating], y: NDArray[np.floating], weights: NDArray[np.floating],
                 max_width: Optional[float] = None, min_width: float = 0.0,
                 params: Optional[Dict[str, Any]] = None):
        self.x = _as_1d(x)
        self.y = _as_1d(y)
        _check_xy(self.x, self.y, "DP_WeightedLine")

        self.weights = _as_1d(weights)
        if self.weights.size != self.x.size - 1:
            raise ValueError("DP_WeightedLine: weights must have length len(x) - 1 (one per segment)")

        base_params = _copy_params(params)
        if max_width is None:
            max_width = float(base_params.get("linewidth", base_params.get("lw", 1.0)))

        self.min_width = float(min_width)
        self.max_width = float(max_width)

        # Build a collection of small DP_Line segments
        collection = DP_Collection()
        w_max = float(np.max(self.weights))
        w_min = float(np.min(self.weights))
        denom = (w_max - w_min)

        for i in range(self.x.size - 1):
            # Compute width safely
            if denom <= 0.0:
                width_i = self.max_width
            else:
                width_i = self.min_width + (self.weights[i] - w_min) / denom * (self.max_width - self.min_width)

            params_i = dict(base_params)
            params_i["linewidth"] = width_i
            if i > 0:
                params_i.pop("label", None)

            collection.add(
                DP_Line(
                    x=np.array([self.x[i], self.x[i + 1]], dtype=float),
                    y=np.array([self.y[i], self.y[i + 1]], dtype=float),
                    params=params_i,
                )
            )

        self._inner = collection

    def _matplotlib_plot(self, ax: plt.Axes) -> None:
        self._inner._matplotlib_plot(ax)

    def _plotly_plot(self, fig: go.Figure, *, row: Optional[int] = None, col: Optional[int] = None) -> None:
        self._inner._plotly_plot(fig, row=row, col=col)
