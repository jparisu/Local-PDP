"""Convert Faex plotter params to Matplotlib kwargs."""

import logging

from typing import Any, Dict, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from faex.plotting.plotter_params import LineParams, ScatterParams, HistParams, AreaParams, ErrorBarParams
from faex.plotting.plotter_symbols import SCATTER_SYMBOLS

logger = logging.getLogger(__name__)

@dataclass
class MatplotlibAxes:
    ax : plt.Axes
    fig : plt.Figure

    def show(self) -> None:
        self.fig.show()


def mlp_symbol(symbol: str) -> str:
    """
    Check if the provided symbol is valid for Matplotlib.
    If not, try to map from plotly to matplotlib.
    Otherwise fails.
    """
    for plotly_sym, mpl_sym in SCATTER_SYMBOLS:
        if symbol == mpl_sym:
            return mpl_sym
        if symbol == plotly_sym:
            return mpl_sym

    raise ValueError(f"Symbol '{symbol}' not valid for Matplotlib.")



def mpl_line_kwargs(p: LineParams) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if p.color is not None:
        kw["color"] = p.color
    if p.opacity is not None:
        kw["alpha"] = p.opacity
    if p.alpha is not None:
        kw["alpha"] = p.alpha
    if p.linewidth is not None:
        kw["linewidth"] = p.linewidth
    if p.style is not None:
        kw["linestyle"] = p.style
    if p.marker is not None:
        kw["marker"] = mlp_symbol(p.marker)
    if p.markersize is not None:
        kw["markersize"] = p.markersize
    if p.zorder is not None:
        kw["zorder"] = p.zorder
    if p.label is not None:
        kw["label"] = p.label
    return kw


def mpl_scatter_kwargs(p: ScatterParams) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if p.color is not None:
        kw["c"] = p.color
    if p.opacity is not None:
        kw["alpha"] = p.opacity
    if p.alpha is not None:
        kw["alpha"] = p.alpha
    if p.size is not None:
        kw["s"] = p.size
    if p.marker is not None:
        kw["marker"] = mlp_symbol(p.marker)
    if p.edgecolor is not None:
        kw["edgecolors"] = p.edgecolor
    if p.edgewidth is not None:
        kw["linewidths"] = p.edgewidth
    if p.cmap is not None:
        kw["cmap"] = p.cmap
    if p.vmin is not None:
        kw["vmin"] = p.vmin
    if p.vmax is not None:
        kw["vmax"] = p.vmax
    if p.zorder is not None:
        kw["zorder"] = p.zorder
    if p.label is not None:
        kw["label"] = p.label
    return kw


def mpl_hist_kwargs(p: HistParams) -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if p.bins is not None:
        kw["bins"] = p.bins
    if p.range is not None:
        kw["range"] = p.range
    if p.density is not None:
        kw["density"] = p.density
    if p.orientation is not None:
        kw["orientation"] = p.orientation  # 'vertical' or 'horizontal'

    if p.color is not None:
        kw["color"] = p.color
    if p.edgecolor is not None:
        kw["edgecolor"] = p.edgecolor
    if p.linewidth is not None:
        kw["linewidth"] = p.linewidth
    if p.zorder is not None:
        kw["zorder"] = p.zorder
    if p.label is not None:
        kw["label"] = p.label

    if p.opacity is not None:
        kw["alpha"] = p.opacity
    if p.alpha is not None:
        kw["alpha"] = p.alpha

    return kw


def mpl_area_kwargs(p: AreaParams) -> Dict[str, Any]:
    """
    For Matplotlib you'll typically call:
        ax.fill_between(p.x, p.y, **mpl_area_kwargs(p))
    """
    kw: Dict[str, Any] = {}
    if p.color is not None:
        kw["color"] = p.color
    # choose which opacity to use for the fill
    alpha = p.fill_opacity if p.fill_opacity is not None else p.opacity
    alpha = alpha if alpha is not None else p.alpha
    if alpha is not None:
        kw["alpha"] = alpha
    if p.linewidth is not None:
        kw["linewidth"] = p.linewidth
    if p.style is not None:
        kw["linestyle"] = p.style
    if p.edgecolor is not None:
        kw["edgecolor"] = p.edgecolor
    if p.zorder is not None:
        kw["zorder"] = p.zorder
    if p.label is not None:
        kw["label"] = p.label

    return kw


def mpl_errorbar_kwargs(p: ErrorBarParams) -> Dict[str, Any]:
    """
    For Matplotlib you'll typically call:
        ax.errorbar(x, y, **mpl_errorbar_kwargs(p))
    """
    kw: Dict[str, Any] = {}

    # Visual appearance
    if p.color is not None:
        # Matplotlib uses 'ecolor' for error bar color
        kw["ecolor"] = p.color

    alpha = p.opacity if p.opacity is not None else p.alpha
    if alpha is not None:
        kw["alpha"] = alpha

    if p.elinewidth is not None:
        kw["elinewidth"] = p.elinewidth
    if p.capsize is not None:
        kw["capsize"] = p.capsize
    if p.capthick is not None:
        kw["capthick"] = p.capthick

    # Z-order / label
    if p.zorder is not None:
        kw["zorder"] = p.zorder
    if p.label is not None:
        kw["label"] = p.label

    return kw



def generate_default_double_figure_matplotlib(
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
        *,
        x_label: str = "Feature",
        y_label: str = "Predictions",
        y2_label: str = "Distributions",
        title: str = "Explanations",
        padding: float = 0.02,
) -> Tuple[MatplotlibAxes, MatplotlibAxes]:
    """
    Generate a default matplotlib figure with two subplots: one for the main plot and one for the distribution.

    Returns:
        Tuple[plt.Figure, List[plt.Axes]]: The created figure and a list of axes [top_axis, bottom_axis].
    """

    logger.debug(f"Generating default double figure with x_limits={x_limits}, y_limits={y_limits}")

    fig = plt.figure(figsize=(12,6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[8, 2])
    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

    # Configure figure
    fig.suptitle(title)

    # Set X axis
    x_min_value = x_limits[0] - (x_limits[1] - x_limits[0]) * padding
    x_max_value = x_limits[1] + (x_limits[1] - x_limits[0]) * padding
    ax_top.set_xlim(x_min_value, x_max_value)
    ax_bottom.set_xlim(x_min_value, x_max_value)
    ax_bottom.set_xlabel(x_label)

    # Set Top axis
    min_y_value = y_limits[0] - (y_limits[1] - y_limits[0]) * padding
    max_y_value = y_limits[1] + (y_limits[1] - y_limits[0]) * padding
    ax_top.set_ylabel(y_label)
    ax_top.set_ylim(min_y_value, max_y_value)
    ax_top.grid()
    ax_top.set_xticklabels([])

    # Set Bottom axis
    min_y2_value = 0
    max_y2_value = 1 + 10 * padding
    ax_bottom.set_ylabel(y2_label)
    ax_bottom.grid()
    ax_bottom.set_ylim(min_y2_value, max_y2_value)
    # Remove number values in bottom axis
    ax_bottom.set_yticklabels([])

    # Activate top axis legend to top-left
    # TODO : enable when needed
    # dummy = ax_top.plot([], [], label="")[0]  # invisible artist
    # ax_top.legend(loc='upper left')

    return MatplotlibAxes(ax_top, fig), MatplotlibAxes(ax_bottom, fig)
