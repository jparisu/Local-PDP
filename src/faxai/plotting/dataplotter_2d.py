
from __future__ import annotations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from matplotlib.collections import LineCollection

from faxai.plotting.DataPlotter import DataPlotter
from faxai.data.DataHolder import DataHolder


class DP_Line(DataPlotter):
    """
    Data Plotter for Line Plots
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
    Data Plotter for Scatter Plots
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
            self.params['bins'] = bins

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

    def __init__(self, data: list[DataPlotter] = None, params: dict = None, axis: int = 0):
        super().__init__(axis)
        self.data = data if data else []
        self.params = dict(params) if params else {}

    def add(self, item: DataPlotter) -> None:
        self.data.append(item)

    def matplotlib_plot(self, ax: plt.axes) -> None:
        for item in self.data:
            item.matplotlib_plot(ax)

    def from_dataholder(
            cls,
            dataholder: DataHolder, params: dict = None, axis: int = 0) -> DP_Collection:
        """
        Create a DP_Collection from a DataHolder.

        Args:
            dataholder (DataHolder): The DataHolder to create the DP_Collection from.
            params (dict): Parameters for the plot.
            axis (int): Axis to plot on.

        Returns:
            DP_Collection: The created DP_Collection.
        """
        params = dict(params) if params else {}
        plotter = cls(params=params, axis=axis)

        if dataholder.ndim() != 2:
            raise ValueError("DataHolder must be 2 dimensional to create a DP_Collection.")

        for i in range(dataholder.shape()[1]):
            y = dataholder.get_column(i)
            plotter.add(
                DP_Line(
                    x=dataholder.get_column(0),
                    y=y,
                    params=params,
                    axis=axis
                )
            )

        return plotter


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
