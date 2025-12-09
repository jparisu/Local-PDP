from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import logging

from faex.plotting.plotter_matplotlib import MatplotlibAxes
from faex.plotting.plotter_plotly import PlotlyAxes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DataPlotter(ABC):
    """
    Base interface for lightweight plotting adapters that can render to
    both Matplotlib and Plotly.
    """

    def matplotlib_plot(self, axis: MatplotlibAxes | None = None) -> MatplotlibAxes:
        """
        Render this plotter onto a Matplotlib Axes.
        If `ax` is None, creates a new figure and axes.
        """
        if axis is None:
            fig, ax = plt.subplots()
            axis = MatplotlibAxes(ax=ax, fig=fig)

        self._matplotlib_plot(axis)
        return axis

    def plotly_plot(self, axis: PlotlyAxes | None = None) -> PlotlyAxes:
        """
        Render this plotter onto a Plotly Figure.
        If `fig` is None, creates a new figure.
        """
        if axis is None:
            axis = make_subplots()
        self._plotly_plot(axis)
        return axis

    @abstractmethod
    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        """Render this plotter onto a Matplotlib Axes."""

    @abstractmethod
    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        """Render this plotter onto a Plotly Figure (optionally to a given subplot)."""

    def _deactivate_label(self) -> None:
        """TODO"""
        ...


class DP2_Empty(DataPlotter):
    """An empty plotter that draws nothing."""

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        pass

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        pass


class DP2_Collection(DataPlotter):
    """A container for other plotters."""

    def __init__(self, plotters: Iterable[DataPlotter] | None = None):
        self.plotters: List[DataPlotter] = list(plotters) if plotters is not None else []

    def add(self, plotter: DataPlotter) -> None:
        """Add a plotter to this collection."""
        self.plotters.append(plotter)

    def extend(self, plotters: Iterable[DataPlotter]) -> None:
        """Add multiple plotters to this collection."""
        self.plotters.extend(plotters)

    def _matplotlib_plot(self, axis: MatplotlibAxes) -> None:
        logger.debug(f"Plotting DP2_Collection with {len(self.plotters)} plotters to Matplotlib.")
        for plotter in self.plotters:
            plotter._matplotlib_plot(axis)

    def _plotly_plot(self, axis: PlotlyAxes) -> None:
        logger.debug(f"Plotting DP2_Collection with {len(self.plotters)} plotters to Plotly.")
        for plotter in self.plotters:
            plotter._plotly_plot(axis)
