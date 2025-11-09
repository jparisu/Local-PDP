"""
Lightweight wrapper around Python's `random.Random` with a stable seeding API.
"""

from __future__ import annotations

import random
from collections.abc import MutableSequence, Sequence
from typing import TypeVar

T = TypeVar("T")


class RandomGenerator:
    """
    A simple wrapper around :class:`random.Random` that provides a few
    convenience methods and stable seed (re)initialization.

    The instance keeps track of the *initial* seed used at construction so you
    can return to it later via :meth:`reset_seed`.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the random number generator.

        Args:
            seed: Seed for the RNG. If ``None``, a seed is sampled uniformly
                from ``[0, 2**32 - 1]`` using :func:`random.randint`.

        Notes:
            The actual seed used is stored and becomes the target of
            :meth:`reset_seed`.
        """
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        self._initial_seed = seed
        self._seed = seed
        self._rng = random.Random(seed)

    def rand(self) -> float:
        """
        Generate a random float in the half-open interval ``[0.0, 1.0)``.

        Returns:
            A float sampled uniformly from ``[0.0, 1.0)``.
        """
        return self._rng.random()

    # Alias for rand
    random = rand

    def randint(self, low: int, high: int) -> int:
        """
        Generate an integer ``N`` such that ``low <= N < high``.

        This is equivalent to :meth:`random.Random.randrange` with ``start=low``
        and ``stop=high``.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).

        Returns:
            An integer in ``[low, high)``.

        Raises:
            ValueError: If ``low >= high``.
        """
        return self._rng.randrange(low, high)

    def choice(self, seq: Sequence[T]) -> T:
        """
        Choose a random element from a non-empty sequence.

        Args:
            seq: A non-empty sequence (e.g., list, tuple, str).

        Returns:
            A single element of ``seq``.

        Raises:
            IndexError: If ``seq`` is empty.
        """
        return self._rng.choice(seq)

    def set_seed(self, seed: int) -> None:
        """
        Set the RNG seed.

        Args:
            seed: New seed for the random number generator.
        """
        self._seed = seed
        self._rng = random.Random(seed)

    def reset_seed(self) -> None:
        """
        Reset the RNG to the seed used at initialization.
        """
        self.set_seed(self._initial_seed)

    def shuffle(self, seq: MutableSequence[T]) -> None:
        """
        Shuffle a mutable sequence *in place*.

        Args:
            seq: A mutable sequence (e.g., a ``list``) to be shuffled.

        Returns:
            None. The input sequence is modified in place.
        """
        self._rng.shuffle(seq)

    def integers(self, low: int, high: int, n: int = 1) -> list[int]:
        """
        Generate samples of random integers in the range ``[low, high)``.

        Args:
            low: Lower bound (inclusive).
            high: Upper bound (exclusive).
            n: Number of samples to generate.

        Returns:
            A list of ``n`` integers drawn uniformly from ``[low, high)``.
        """
        return [self._rng.randint(low, high - 1) for _ in range(n)]

    def gauss(self, mean: float, std: float, n: int = 1) -> list[float]:
        """
        Generate samples from a normal (Gaussian) distribution.

        Args:
            mean: Mean of the normal distribution.
            std: Standard deviation of the normal distribution.
            n: Number of samples to generate.

        Returns:
            A list of ``n`` samples drawn from the specified normal distribution.
        """
        return [self._rng.gauss(mean, std) for _ in range(n)]

    def uniform(self, low: float, high: float, n: int = 1) -> list[float]:
        """
        Generate samples from a uniform distribution over ``[a, b)``.

        Args:
            low: Lower bound of the uniform distribution (inclusive).
            high: Upper bound of the uniform distribution (exclusive).
            n: Number of samples to generate.

        Returns:
            A list of ``n`` samples drawn from the specified uniform distribution.
        """
        return [self._rng.uniform(low, high) for _ in range(n)]
