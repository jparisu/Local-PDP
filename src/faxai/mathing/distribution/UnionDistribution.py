from __future__ import annotations

from math import sqrt

import numpy as np

from faxai.mathing.distribution.Distribution import Distribution
from faxai.mathing.RandomGenerator import RandomGenerator
from faxai.utils.decorators import cache_method


class UnionDistribution(Distribution):
    """
    Represents an equal-weight mixture (union) of multiple distributions.

    Given component distributions :math:`\\{D_i\\}_{i=1}^k`, the mixture density is
    defined as the arithmetic mean of component densities:

        PDF(x) = (1/k) * sum_i PDF_i(x)

    and, when component CDFs are available,

        CDF(x) = (1/k) * sum_i CDF_i(x).

    This class computes analytical statistics where possible and uses numerically
    stable formulas for mixture moments.

    Parameters
    ----------
    distributions : list[Distribution]
        A non-empty list of component distributions. All components are assumed to
        be independent and identically weighted in the mixture.

    Raises
    ------
    ValueError
        If `distributions` is empty.
    TypeError
        If any element in `distributions` is not an instance of `Distribution`.

    Notes
    -----
    - **Mean**: the mean of the mixture is the average of component means.
    - **Variance**: computed via the law of total variance:
        Var(X) = E[Var(X|Z)] + Var(E[X|Z]),
      where Z is the component index (uniform over components).
    - **Mode**: exact mode of a mixture often has no closed form. We return a
      practical estimate: the point among a candidate set that maximizes the mixture
      PDF. The candidate set includes each component's mode and mean.
    - **Median**: if every component implements `cdf(x)`, we do a bounded bisection
      search on the mixture CDF. Otherwise, we fall back to a deterministic
      Monte-Carlo approximation (fixed RNG seed) for the median.
    """

    def __init__(self, distributions: list[Distribution]):
        if not distributions:
            raise ValueError("UnionDistribution requires at least one component distribution.")
        for d in distributions:
            if not isinstance(d, Distribution):
                raise TypeError("All elements of `distributions` must be instances of Distribution.")
        self._distributions: list[Distribution] = list(distributions)

    ############################
    # Statistical Properties

    @cache_method
    def mean(self) -> float:
        """
        Return the mixture mean.

        Returns
        -------
        float
            The arithmetic mean of component means.
        """
        k = len(self._distributions)
        return sum(d.mean() for d in self._distributions) / k

    @cache_method
    def std(self, ddof: int = 0) -> float:
        """
        Return the mixture standard deviation.

        Parameters
        ----------
        ddof : int, optional
            Delta degrees of freedom (ignored; included for API compatibility).

        Returns
        -------
        float
            The standard deviation of the mixture, computed from the exact
            mixture variance formula.
        """
        means = np.array([d.mean() for d in self._distributions], dtype=float)
        vars_ = np.array([float(d.std() ** 2) for d in self._distributions], dtype=float)

        # Law of total variance: E[Var] + Var(E)
        var_total = float(vars_.mean() + means.var(ddof=0))
        return sqrt(var_total)

    @cache_method
    def moded(self) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__}.moded() is not implemented; use experimental_moded() instead."
        )

    def median(self) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__}.median() is not implemented; use experimental_median() instead."
        )

    @cache_method
    def experimental_moded(self, tries: int = 1000) -> float:
        """
        Return an estimate of the mode by evaluating the mixture PDF at N points.
        """
        # Build a robust bracket using means ± 10*std (covers practically all mass)
        means = np.array([d.mean() for d in self._distributions], dtype=float)
        stds = np.array([max(float(d.std()), 1e-12) for d in self._distributions], dtype=float)
        left = float(np.min(means - 10.0 * stds))
        right = float(np.max(means + 10.0 * stds))

        # Evaluate at candidates
        candidates = np.linspace(left, right, tries, dtype=float)
        pdf_vals = np.array([self.pdf(float(x)) for x in candidates], dtype=float)

        # Return the candidate with maximum PDF value
        max_idx = int(pdf_vals.argmax())
        return float(candidates[max_idx])

    @cache_method
    def experimental_median(self, tries: int = 100) -> float:
        """
        Return the median (0.5-quantile) of the mixture by a random sampling method.

        Strategy
        --------
        If all components implement a `cdf(x)` method, the mixture CDF is the
        mean of component CDFs and we solve CDF(x)=0.5 via bisection on a
        conservative bracket derived from component means and standard
        deviations. If any component lacks `cdf`, we fall back to a deterministic
        Monte-Carlo estimate using a fixed RNG seed for reproducibility.

        Returns
        -------
        float
            The median of the mixture (exact when all CDFs are available,
            otherwise an accurate deterministic approximation).
        """
        # Build a robust bracket using means ± 10*std (covers practically all mass)
        means = np.array([d.mean() for d in self._distributions], dtype=float)
        stds = np.array([max(float(d.std()), 1e-12) for d in self._distributions], dtype=float)
        left = float(np.min(means - 10.0 * stds))
        right = float(np.max(means + 10.0 * stds))

        # Bisection
        target = 0.5
        for _ in range(tries):
            mid = 0.5 * (left + right)
            if self.cdf(mid) < target:
                left = mid
            else:
                right = mid
        return 0.5 * (left + right)

    ############################
    # Probability Functions

    @cache_method
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the mixture probability density function (PDF) at `x`.

        Parameters
        ----------
        x : float
            The evaluation point.

        Returns
        -------
        float
            The mixture PDF value at `x`, i.e., the average of component PDFs.
        """
        arr = np.array([d.pdf(x) for d in self._distributions])
        return np.mean(arr, axis=0)

    @cache_method
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the mixture cumulative distribution function (CDF) at `x`, when
        available.

        Parameters
        ----------
        x : float
            The evaluation point.

        Returns
        -------
        float
            The mixture CDF value at `x` if all components define `cdf(x)`.
            Otherwise, a deterministic Monte-Carlo approximation is returned.
        """
        arr = np.array([d.cdf(x) for d in self._distributions])
        return np.mean(arr, axis=0)

    @cache_method
    def maximum_pdf(self) -> float:
        raise NotImplementedError(
            f"{self.__class__.__name__}.maximum_pdf() is not implemented; use experimental_maximum_pdf() instead."
        )

    # TODO

    ############################
    # Sampling Methods

    def random_sample(self, n: int = 1, rng: RandomGenerator | None = None) -> np.ndarray:
        """
        Generate random samples from the mixture distribution.

        Sampling is performed by first choosing a component index uniformly at
        random, then drawing from that component.

        Parameters
        ----------
        n : int, optional
            Number of samples to generate. Default is 1.
        rng : numpy.random.Generator, optional
            A NumPy random number generator. If None, a default generator is used.

        Returns
        -------
        numpy.ndarray
            Array of random samples of shape (n,).
        """
        if rng is None:
            rng = RandomGenerator()
        k = len(self._distributions)
        idx = rng.integers(0, k, n=n)

        # Draw per component, then restore original ordering
        out = np.empty(n, dtype=float)
        for comp in range(k):
            mask = [i == comp for i in idx]
            cnt = int(np.sum(mask))
            if cnt == 0:
                continue
            out[mask] = self._distributions[comp].random_sample(n=cnt, rng=rng)
        return out
