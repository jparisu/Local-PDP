from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import faxai.data.DataHolder as dh
import faxai.data.DataPlotter as dp

def from_hyperplane_to_line(
    hyperplane: dh.HyperPlane,
    params: Optional[Dict[str, Any]] = None,
) -> dp.DP_Line:
    """
    Convert a Hyperplane DataHolder to a DP_Line DataPlotter.
    """
    # Check the hyperplane is 1d
    if len(hyperplane.shape()) != 1:
        raise ValueError(
            "Hyperplane must be 1-dimensional to convert to DP_Line."
        )

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
    if len(hyperplanes.shape()) != 2:
        raise ValueError(
            "All Hyperplanes must be 1-dimensional to convert to DP_Lines."
        )

    plotter = dp.DP_Collection(params=params)

    # Create a line for each hyperplane, that is first dimension in ndarray in values
    for hp in hyperplanes.it_hyperplanes():

        line = from_hyperplane_to_line(
            hyperplane=hp,
            params=params,
        )

        plotter.add(line)

        if 'label' in params:
            params = dict(params)
            params.pop('label')

    return plotter
