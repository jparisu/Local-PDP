"""
2D Plotting Parameters Data Classes
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Union, TypeVar, Type
from copy import deepcopy

import numpy as np

logger = logging.getLogger(__name__)

ArrayLike = np.ndarray[Any]

T = TypeVar("T", bound="PlotParams")

from dataclasses import fields, is_dataclass


class PlotParams:

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        instance = cls()
        return instance.update_from_dict(data)

    def copy(self: T) -> T:
        return deepcopy(self)

    def set_default_value(self, field_name: str, default_value: Any) -> None:
        if getattr(self, field_name) is None:
            setattr(self, field_name, default_value)

    def update_from_dict(self: T, data: dict) -> T:
        # All valid dataclass field names
        valid_fields = {f.name for f in fields(self)}

        for key, value in data.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                logger.warning(f"Setting key '{key}' not in dataclass '{type(self).__name__}'")

        return self


# ---------- LINE ----------
@dataclass
class LineParams(PlotParams):
    color: Optional[str] = None
    opacity: Optional[float] = None        # 0..1
    alpha: Optional[float] = None        # 0..1, alias for opacity
    linewidth: Optional[float] = None
    style: Optional[str] = None            # '-', '--', ':', '-.', 'none'
    marker: Optional[str] = None
    markersize: Optional[float] = None
    zorder: Optional[int] = None
    label: Optional[str] = None


# ---------- SCATTER ----------
@dataclass
class ScatterParams(PlotParams):
    color: Optional[str] = None            # or array-like for colormap
    opacity: Optional[float] = None
    alpha: Optional[float] = None        # 0..1, alias for opacity
    size: Optional[Union[float, ArrayLike]] = None
    marker: Optional[str] = None           # 'o', 's', '^', etc.
    edgecolor: Optional[str] = None
    edgewidth: Optional[float] = None
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    zorder: Optional[int] = None
    label: Optional[str] = None


# ---------- HISTOGRAM ----------
@dataclass
class HistParams(PlotParams):
    bins: Optional[Union[int, Sequence[float]]] = None
    range: Optional[Sequence[float]] = None
    binwidth: Optional[float] = None       # used more in Plotly; mpl: can turn into bins
    density: Optional[bool] = None
    orientation: Optional[str] = None      # 'vertical' / 'horizontal' | 'v' / 'h'

    color: Optional[str] = None
    edgecolor: Optional[str] = None
    linewidth: Optional[float] = None
    opacity: Optional[float] = None
    alpha: Optional[float] = None        # 0..1, alias for opacity
    zorder: Optional[int] = None
    label: Optional[str] = None

    # TODO
    # relative: bool = False   # normalize by max so heights [0, 1]
    # max_height: Optional[float] = None  # final bar height is scaled so max = this


# ---------- AREA ----------
@dataclass
class AreaParams(PlotParams):
    color: Optional[str] = None
    opacity: Optional[float] = None        # overall opacity
    alpha: Optional[float] = None        # 0..1, alias for opacity
    fill_opacity: Optional[float] = None   # separate fill alpha if you want
    linewidth: Optional[float] = None
    style: Optional[str] = None
    edgecolor: Optional[str] = None
    zorder: Optional[int] = None
    label: Optional[str] = None


# ---------- ERROR BARS ----------
@dataclass
class ErrorBarParams(PlotParams):
    # Visual appearance
    color: Optional[str] = None              # Color of error bar lines
    opacity: Optional[float] = None          # 0..1
    alpha: Optional[float] = None            # alias for opacity
    elinewidth: Optional[float] = None       # Error bar line width
    capsize: Optional[float] = None          # Length of error bar caps
    capthick: Optional[float] = None         # Thickness of cap line

    # Z-order / label
    zorder: Optional[int] = None
    label: Optional[str] = None
