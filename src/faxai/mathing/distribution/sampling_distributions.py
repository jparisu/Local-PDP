from __future__ import annotations

import numpy as np

from faxai.mathing.distribution.Distribution import Distribution


class DeltaDistribution(Distribution):
    """
    Represents a degenerate distribution concentrated only on those points that are given as samples.
    """

    def __init__(self, samples: np.ndarray): ...


class HistogramDistribution(Distribution):
    """
    Represents a discrete distribution based on a given set of samples.
    Probability functions are set using histogram estimation.
    """

    def __init__(self, samples: np.ndarray, bins: int | None = 10, bins_range: float | None = None): ...


class WeightedDistribution(Distribution):
    """
    Represents a distribution based on a given set of samples with an associated weight.
    Probability functions are set using histogram estimation.
    """

    def __init__(
        self, samples: np.ndarray, weights: np.ndarray, bins: int | None = 10, bins_range: float | None = None
    ): ...
