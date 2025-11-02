from __future__ import annotations

import numpy as np

from faxai.mathing.distribution.Distribution import Distribution
from faxai.mathing.kernel import Kernel


class KernelDensityDistribution(Distribution):
    """
    Represents a distribution given by a kernel function.
    """

    def __init__(self, kernel: Kernel):
        # TODO
        raise NotImplementedError


class KernelDensityEstimationDistribution(Distribution):
    """
    Represents a distribution estimated from a sample using Kernel Density Estimation (KDE).
    """

    def __init__(self, samples: np.ndarray, kernel: Kernel):
        # TODO
        raise NotImplementedError
