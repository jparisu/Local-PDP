from __future__ import annotations

from typing import Any, Dict, Optional, Callable
import logging
import numpy as np

import faex.data.DataHolder as dh
import faex.plotting.DataPlotter as dp
import faex.plotting.d2.dataplotter_primitives_2d as dp2
import faex.plotting.d2.dataplotter_special_2d as dps2
from faex.plotting.plotter_params import AreaParams

logger = logging.getLogger(__name__)


########################################################################################################################
# LINES

def to_lines(
    data: dh.DataHolder,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP2_Collection | dp2.DP2_Line:
    """
    Convert a DataHolder to a DP2_Line DataPlotter.
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
) -> dp2.DP2_Line:
    """
    Convert a Hyperplane DataHolder to a DP2_Line DataPlotter.
    """
    # Check the hyperplane is 1d
    if len(hyperplane.shape()) < 1:
        # Empty HyperPlane, nothing to plot
        return dp.DP2_Empty()

    if len(hyperplane.shape()) > 1:
        raise ValueError("Hyperplane must be 1-dimensional to convert to DP2_Line.")

    coords = hyperplane.grid
    values = hyperplane.target

    return dp2.DP2_Line(
        x=coords[0],
        y=values,
        params=params,
    )


def from_hyperplane_to_area_line(
    hyperplane: dh.HyperPlane,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP2_Collection:
    """
    Convert a Hyperplane DataHolder to a line and the area under it.
    """

    params = dict(params) if params else {}

    # Check the hyperplane is 1d
    if len(hyperplane.shape()) < 1:
        # Empty HyperPlane, nothing to plot
        return dp.DP2_Empty()

    if len(hyperplane.shape()) > 1:
        raise ValueError("Hyperplane must be 1-dimensional to convert to DP2_Line.")

    coords = hyperplane.grid
    values = hyperplane.target

    line = dp2.DP2_Line(
        x=coords[0],
        y=values,
        params=params,
    )

    # If alpha set, reduce in half. If not, set 0.5
    if "opacity" in params:
        area_alpha = params["opacity"] / 2
    else:
        area_alpha = 0.5
    params = dict(params)
    params["opacity"] = area_alpha

    area = dp2.DP2_Area(
        x=coords[0],
        y_min=np.zeros_like(values),
        y_max=values,
        params=params,
    )

    return dp.DP2_Collection([line, area])


def from_hyperplanes_to_lines(
    hyperplanes: dh.HyperPlanes,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP2_Collection:
    """
    Convert a list of Hyperplane DataHolders to a DP2_Lines DataPlotter.
    """

    params = dict(params) if params else {}

    # Check the hyperplanes are 1d
    if len(hyperplanes.shape()) < 2:
        # Empty HyperPlanes, nothing to plot
        return dp.DP2_Empty()

    if len(hyperplanes.shape()) > 2:
        raise ValueError("All Hyperplanes must be 1-dimensional to convert to DP2_Lines.")

    plotter = dp.DP2_Collection()

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
) -> dp.DP2_Collection:
    """
    Convert a DataHolderCollection to a DP2_Lines DataPlotter.
    """

    params = dict(params) if params else {}

    plotter = dp.DP2_Collection()

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
# WEIGHTED LINES


def from_weighted_hyperplane_to_line(
    w_hyperplane: dh.WeightedHyperPlane,
    params: Optional[Dict[str, Any]] = None,
    type_of_line: Callable[..., dp.DataPlotter]  = dps2.DP2_WeightedOpacityLine,
) -> dp.DataPlotter:
    """
    Convert a Hyperplane DataHolder to a DP2_Line DataPlotter.
    """
    # Check the hyperplane is 1d
    if len(w_hyperplane.shape()) < 1:
        # Empty HyperPlane, nothing to plot
        return dp.DP2_Empty()

    if len(w_hyperplane.shape()) > 1:
        raise ValueError("Hyperplane must be 1-dimensional to convert to DP2_Line.")

    coords = w_hyperplane.grid
    values = w_hyperplane.target
    weights = w_hyperplane.weights

    return type_of_line(
        x=coords[0],
        y=values,
        weights=weights,
        params=params,
    )


def from_weighted_hyperplanes_to_lines(
    w_hyperplanes: dh.WeightedHyperPlanes,
    params: Optional[Dict[str, Any]] = None,
    type_of_line: Callable[..., dp.DataPlotter]  = dps2.DP2_WeightedOpacityLine,
) -> dp.DataPlotter:
    """
    Convert a list of Hyperplane DataHolders to a DP2_Lines DataPlotter.
    """

    params = dict(params) if params else {}

    # Check the hyperplanes are 1d
    if len(w_hyperplanes.shape()) < 2:
        # Empty HyperPlanes, nothing to plot
        return dp.DP2_Empty()

    if len(w_hyperplanes.shape()) > 2:
        raise ValueError("All Hyperplanes must be 1-dimensional to convert to DP2_Lines.")

    plotter = dp.DP2_Collection()

    # Create a line for each hyperplane, that is first dimension in ndarray in values
    for whp in w_hyperplanes.it_weighted_hyperplanes():
        line = from_weighted_hyperplane_to_line(
            w_hyperplane=whp,
            params=params,
            type_of_line=type_of_line,
        )

        plotter.add(line)

        if "label" in params:
            params = dict(params)
            params.pop("label")

    return plotter

########################################################################################################################
# SCATTER

def from_hyperplanes_to_scatter(
    hyperplanes: dh.HyperPlanes,
    params: Optional[Dict[str, Any]] = None,
) -> dp2.DP2_Scatter:
    """
    Convert a list of Hyperplane DataHolders to a DP2_Scatter DataPlotter.
    """

    params = dict(params) if params else {}

    # Check the hyperplanes are 1d
    if len(hyperplanes.grid.shape()) != 1:
        raise ValueError("Hyperplanes must have 1-dimensional grid to convert to DP2_Scatter.")

    x_values = []
    y_values = []

    for hp in hyperplanes.it_hyperplanes():
        x_values.extend(hp.grid[0])
        y_values.extend(hp.target)

    return dp2.DP2_Scatter(
        x=np.array(x_values),
        y=np.array(y_values),
        params=params,
    )


########################################################################################################################
# AREA

def from_hyperplane_to_area_under_curve(
    hyperplane: dh.HyperPlane,
    params: Optional[Dict[str, Any]] = None,
) -> dp2.DP2_Area:
    """
    Convert a Hyperplane DataHolder to a DP2_Line DataPlotter.
    """
    # Check the hyperplane is 1d
    if len(hyperplane.shape()) < 1:
        # Empty HyperPlane, nothing to plot
        return dp.DP2_Empty()

    if len(hyperplane.shape()) > 1:
        raise ValueError("Hyperplane must be 1-dimensional to convert to DP2_Line.")

    x = hyperplane.grid
    y_max = hyperplane.target
    # Set zeros
    y_min = np.zeros_like(y_max)

    return dp2.DP2_Area(
        x=x,
        y_min=y_min,
        y_max=y_max,
        params=params,
    )


def from_distributions_to_area(
    distributions: dh.DistributionCollection,
    params: AreaParams,
    sigmas = 2,
) -> dp2.DP2_Collection:
    """
    Convert a DistributionCollection DataHolder to a DP2_Area DataPlotter.
    """

    # Check the distributions are 1d
    if len(distributions.shape()) < 1:
        # Empty Distributions, nothing to plot
        raise ValueError("All Distributions must be 1-dimensional to convert to DP2_Area.")

    if len(distributions.shape()) > 1:
        raise ValueError("All Distributions must be 1-dimensional to convert to DP2_Area.")

    # Compute the mean distribution at each grid point
    mean_values = np.array([d.mean() for d in distributions.distributions], dtype=float)
    std_values = np.array([d.std() for d in distributions.distributions], dtype=float)
    x = distributions.grid[0]

    logger.debug(f"X values: {x.shape}")
    logger.debug(f"Mean values: {mean_values.shape}")
    logger.debug(f"Std values: {std_values.shape}")

    # Params must have opacity set for area
    if params.opacity is None:
        params.opacity = 0.5

    plotter = dp.DP2_Collection()

    for s in range(1, sigmas+1):

        # Compute standard distribution higher and lower
        y_max = mean_values + s * std_values
        y_min = mean_values - s * std_values

        logger.debug(f"Sigma level {s}: x={x.shape} y_min={y_min.shape}, y_max={y_max.shape}")

        plotter.add(
            dp2.DP2_Area(
                x=x,
                y_min=y_min,
                y_max=y_max,
                params=params,
            )
        )

        logger.debug(f"Added area for sigma level {s}: y_min={y_min}, y_max={y_max}")

        # Remove label after first
        if params.label is not None:
            params.label = None
        params.opacity = params.opacity / 2

    return plotter



########################################################################################################################
# ERROR BAR

def from_distribution_to_line_with_error_bar(
    distributions: dh.DistributionCollection,
    params: Optional[Dict[str, Any]] = None,
) -> dp2.DP2_ErrorBar:
    """
    Convert a Hyperplane DataHolder to a DP2_ErrorBar DataPlotter.
    """

    # Check the distributions are 1d
    if len(distributions.shape()) < 1:
        # Empty Distributions, nothing to plot
        raise ValueError("All Distributions must be 1-dimensional to convert to DP2_Area.")

    if len(distributions.shape()) > 1:
        raise ValueError("All Distributions must be 1-dimensional to convert to DP2_Area.")

    coords = distributions.grid[0]
    line = np.array([d.mean() for d in distributions.distributions], dtype=float)
    error = np.array([d.mean_confidence_interval() for d in distributions.distributions], dtype=float)

    return dp2.DP2_ErrorBar(
        x=coords,
        y=line,
        error=error,
        params=params,
    )
