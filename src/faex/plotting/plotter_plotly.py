"""Convert Faex plotter params to Matplotlib kwargs."""

from typing import Any, Dict
from dataclasses import dataclass
import plotly.graph_objects as go

from faex.plotting.plotter_params import LineParams, ScatterParams, HistParams, AreaParams, ErrorBarParams
from faex.plotting.plotter_symbols import SCATTER_SYMBOLS

# Type for Plotly axis specification
@dataclass
class PlotlyAxes:
    fig : go.Figure
    row : int = None
    col : int = None

    def add_trace(self, trace: Any) -> None:
        self.fig.add_trace(trace, row=self.row, col=self.col)

    def show(self) -> None:
        self.fig.show()

def plotly_symbol(symbol: str) -> str:
    """
    Check if the provided symbol is valid for Matplotlib.
    If not, try to map from plotly to matplotlib.
    Otherwise fails.
    """
    for plotly_sym, mpl_sym in SCATTER_SYMBOLS:
        if symbol == mpl_sym:
            return plotly_sym
        if symbol == plotly_sym:
            return plotly_sym

    raise ValueError(f"Symbol '{symbol}' not valid for Matplotlib.")


PLOTLY_DASH_MAP = {
    None: None,
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
    "none": None,
}

def plotly_line_kwargs(p: LineParams) -> Dict[str, Any]:
    line: Dict[str, Any] = {}
    marker: Dict[str, Any] = {}
    kw: Dict[str, Any] = {}

    if p.color is not None:
        line["color"] = p.color
    if p.linewidth is not None:
        line["width"] = p.linewidth
    if p.style is not None:
        dash = PLOTLY_DASH_MAP.get(p.style, p.style)
        if dash is not None:
            line["dash"] = dash

    # markers if present
    mode = "lines"
    if p.marker is not None:
        mode = "lines+markers"
        marker["symbol"] = plotly_symbol(p.marker)
    if p.markersize is not None:
        marker["size"] = p.markersize

    kw["mode"] = mode
    if line:
        kw["line"] = line
    if marker:
        kw["marker"] = marker
    if p.opacity is not None:
        kw["opacity"] = p.opacity
    if p.label is not None:
        kw["name"] = p.label

    # Remove from legend if label is None
    if p.label is None:
        kw["showlegend"] = False

    return kw


def plotly_scatter_kwargs(p: ScatterParams) -> Dict[str, Any]:
    marker: Dict[str, Any] = {}
    kw: Dict[str, Any] = {}

    if p.color is not None:
        marker["color"] = p.color
    if p.size is not None:
        marker["size"] = p.size
    if p.marker is not None:
        marker["symbol"] = plotly_symbol(p.marker)
    if p.edgecolor is not None or p.edgewidth is not None:
        marker["line"] = {}
        if p.edgecolor is not None:
            marker["line"]["color"] = p.edgecolor
        if p.edgewidth is not None:
            marker["line"]["width"] = p.edgewidth
    if p.cmap is not None:
        marker["colorscale"] = p.cmap
    if p.vmin is not None or p.vmax is not None:
        # in Plotly you'd typically use coloraxis or normalization outside,
        # but you can store them here for your higher-level API
        kw["_vmin"] = p.vmin
        kw["_vmax"] = p.vmax

    kw["mode"] = "markers"
    if marker:
        kw["marker"] = marker
    if p.opacity is not None:
        kw["opacity"] = p.opacity
    if p.label is not None:
        kw["name"] = p.label

    # Remove from legend if label is None
    if p.label is None:
        kw["showlegend"] = False

    return kw


def plotly_hist_kwargs(p: HistParams) -> Dict[str, Any]:
    marker: Dict[str, Any] = {}
    kw: Dict[str, Any] = {}

    if p.color is not None:
        marker["color"] = p.color
    if p.edgecolor is not None or p.linewidth is not None:
        marker["line"] = {}
        if p.edgecolor is not None:
            marker["line"]["color"] = p.edgecolor
        if p.linewidth is not None:
            marker["line"]["width"] = p.linewidth

    if marker:
        kw["marker"] = marker
    if p.opacity is not None:
        kw["opacity"] = p.opacity
    if p.label is not None:
        kw["name"] = p.label

    # binning
    if p.bins is not None and isinstance(p.bins, int):
        kw["nbinsx"] = p.bins  # or nbinsy if orientation == 'h'
    if p.range is not None:
        kw["xbins"] = {"start": p.range[0], "end": p.range[1]}
    if p.binwidth is not None:
        kw.setdefault("xbins", {})
        kw["xbins"]["size"] = p.binwidth
    # if p.density:
    #     kw["histnorm"] = "probability"

    if p.orientation in ("horizontal", "h"):
        kw["orientation"] = "h"

    # Remove from legend if label is None
    if p.label is None:
        kw["showlegend"] = False

    return kw


def plotly_area_kwargs(p: AreaParams) -> Dict[str, Any]:
    line: Dict[str, Any] = {}
    kw: Dict[str, Any] = {}

    # area is scatter with fill
    if p.color is not None:
        line["color"] = p.color
    if p.linewidth is not None:
        line["width"] = p.linewidth
    if p.style is not None:
        dash = PLOTLY_DASH_MAP.get(p.style, p.style)
        if dash is not None:
            line["dash"] = dash

    kw["mode"] = "lines"
    if line:
        kw["line"] = line
    if p.opacity is not None:
        kw["opacity"] = p.opacity
    if p.label is not None:
        kw["name"] = p.label

    # Remove from legend if label is None
    if p.label is None:
        kw["showlegend"] = False

    return kw


def plotly_errorbar_kwargs(p: ErrorBarParams) -> Dict[str, Any]:
    """
    For Plotly you'll typically call:
        go.Scatter(x=x, y=y, **plotly_errorbar_kwargs(p))
    """
    kw: Dict[str, Any] = {}

    error_x: Dict[str, Any] = {}
    error_y: Dict[str, Any] = {}

    # Shared visual props for error bars
    if p.elinewidth is not None:
        if error_x:
            error_x["thickness"] = p.elinewidth
        if error_y:
            error_y["thickness"] = p.elinewidth

    if p.capsize is not None:
        # Plotly uses 'width' for the width of the end caps (in px)
        if error_x:
            error_x["width"] = p.capsize
        if error_y:
            error_y["width"] = p.capsize

    if p.color is not None:
        if error_x:
            error_x["color"] = p.color
        if error_y:
            error_y["color"] = p.color

    if error_x:
        kw["error_x"] = error_x
    if error_y:
        kw["error_y"] = error_y

    # Trace-level appearance
    if p.color is not None:
        kw["marker"] = {"color": p.color}

    alpha = p.opacity if p.opacity is not None else p.alpha
    if alpha is not None:
        kw["opacity"] = alpha

    # By default, error bars typically go with markers
    kw["mode"] = "markers"

    # Legend handling
    if p.label is not None:
        kw["name"] = p.label
    else:
        kw["showlegend"] = False

    return kw
