from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

import faex.data.DataHolder as dh
import faex.data.DataPlotter as dp


def to_lines(
    data: dh.DataHolder,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Collection | dp.DP_Line:
    """
    Convert a DataHolder to a DP_Line DataPlotter.
    Currently supports only HyperPlane DataHolders.
    """
    if isinstance(data, dh.HyperPlane):
        return from_hyperplane_to_line(
            hyperplane=data,
            params=params,
        )
    elif isinstance(data, dh.HyperPlanes):
        return from_hyperplanes_to_lines(
            hyperplanes=data,
            params=params,
        )
    elif isinstance(data, dh.DataHolderCollection):
        return from_collection_to_lines(
            collection=data,
            params=params,
        )
    else:
        raise ValueError(f"DataHolder of type {type(data)} cannot be converted to lines, or it is not implemented.")


def from_hyperplane_to_line(
    hyperplane: dh.HyperPlane,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Line:
    """
    Convert a Hyperplane DataHolder to a DP_Line DataPlotter.
    """
    # Check the hyperplane is 1d
    if len(hyperplane.shape()) < 1:
        # Empty HyperPlane, nothing to plot
        return dp.DP_Empty()

    if len(hyperplane.shape()) > 1:
        raise ValueError("Hyperplane must be 1-dimensional to convert to DP_Line.")

    coords = hyperplane.grid
    values = hyperplane.target

    return dp.DP_Line(
        x=coords[0],
        y=values,
        params=params,
    )


def from_hyperplanes_to_lines(
    hyperplanes: dh.HyperPlanes,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Collection:
    """
    Convert a list of Hyperplane DataHolders to a DP_Lines DataPlotter.
    """

    params = dict(params) if params else {}

    # Check the hyperplanes are 1d
    if len(hyperplanes.shape()) < 2:
        # Empty HyperPlanes, nothing to plot
        return dp.DP_Empty()

    if len(hyperplanes.shape()) > 2:
        raise ValueError("All Hyperplanes must be 1-dimensional to convert to DP_Lines.")

    plotter = dp.DP_Collection(params=params)

    # Create a line for each hyperplane, that is first dimension in ndarray in values
    for hp in hyperplanes.it_hyperplanes():
        line = from_hyperplane_to_line(
            hyperplane=hp,
            params=params,
        )

        plotter.add(line)

        if "label" in params:
            params = dict(params)
            params.pop("label")

    return plotter


def from_collection_to_lines(
    collection: dh.DataHolderCollection,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Collection:
    """
    Convert a DataHolderCollection to a DP_Lines DataPlotter.
    """

    params = dict(params) if params else {}

    plotter = dp.DP_Collection(params=params)

    for data_holder in collection:
        plotter.add(
            to_lines(
                data=data_holder,
                params=params,
            )
        )

        if "label" in params:
            params = dict(params)
            params.pop("label")

    return plotter


########################################################################################################################


def from_hyperplanes_to_scatter(
    hyperplanes: dh.HyperPlanes,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Scatter:
    """
    Convert a list of Hyperplane DataHolders to a DP_Scatter DataPlotter.
    """

    params = dict(params) if params else {}

    # Check the hyperplanes are 1d
    if len(hyperplanes.grid.shape()) != 1:
        raise ValueError("Hyperplanes must have 1-dimensional grid to convert to DP_Scatter.")

    x_values = []
    y_values = []

    for hp in hyperplanes.it_hyperplanes():
        x_values.extend(hp.grid[0])
        y_values.extend(hp.target)

    return dp.DP_Scatter(
        x=np.array(x_values),
        y=np.array(y_values),
        params=params,
    )
