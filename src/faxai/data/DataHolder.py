from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import numpy as np


class DataHolder(ABC):
    """
    Abstract base class for holding and managing data.

    So far, it is only used for typing purposes.
    """

    def check(self, throw: bool = True) -> bool:
        """
        Abstract method to check if the data is valid.
        Args:
            throw (bool): Whether to throw an exception if the data is invalid.
        Returns:
            bool: True if valid, False otherwise.

        Note:
            It is not required to implement this method, but highly recommended.
        """
        return True

    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """
        Abstract method to get the shape of the data.
        Returns:
            tuple[int, ...]: Shape of the data.
        """
        pass

@dataclass
class Grid(DataHolder):
    """
    A data class representing N arrays to form a N dimensional grid.

    Attributes:
        grid (list[np.ndarray]): List of N arrays representing the grid dimensions.
    """

    grid: list[np.ndarray]

    def check(self, throw: bool = True) -> bool:
        """
        Check if the provided base dimensions are valid for a Grid.
        Returns:
            bool: True if valid, False otherwise.
        """
        # Check base is N dimensional
        if self.grid.ndim < 1:
            if throw:
                raise ValueError("Base must be at least 1 dimensional.")
            return False

        # Check every dimension is a 1D array
        for dimension in self.grid:
            if dimension.ndim != 1:
                if throw:
                    raise ValueError("Every dimension in the grid must be a 1D array.")
                return False

        return True

    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the grid.

        Returns:
            tuple[int, ...]: Shape of the grid.
        """
        return tuple(int(self.grid[i].shape[0]) for i in range(len(self.grid)))


    def __getitem__(self, index: int) -> np.ndarray:
        """
        Get the dimension at the specified index.
        Args:
            index (int): Index of the dimension to retrieve.
        Returns:
            np.ndarray: The dimension at the specified index.
        """
        return self.grid[index]

@dataclass
class HyperPlane(DataHolder):
    """
    A data class representing a N dimensional grid and a target dimension with a value for each point in the grid.

    Attributes:
        grid (Grid): N dimensional base data with shape A1 x A2 x ... x AN.
        target (np.ndarray): matrix with shape (A1, A2, ..., AN) representing the target values for each point in the grid.
    """

    grid: Grid
    target: np.ndarray

    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the hyperplane.
        Returns:
            tuple[int, ...]: Shape of the hyperplane.
        """
        return self.target.shape


@dataclass
class HyperPlanes(DataHolder):
    """
    A data class representing a N dimensional grid and M target dimensions with a value for each point in the grid.

    Attributes:
        grid (Grid): N dimensional base data with shape A1 x A2 x ... x AN.
        targets (np.ndarray): matrix with shape (M, A1, A2, ..., AN) representing the target values for each point in the grid.
    """

    grid: Grid
    targets: np.ndarray

    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the hyperplane.
        Returns:
            tuple[int, ...]: Shape of the hyperplane.
        """
        return self.targets.shape

    def __len__(self) -> int:
        """
        Get the number of hyperplanes.
        Returns:
            int: Number of hyperplanes.
        """
        return self.targets.shape[0]

    def it_hyperplanes(self) -> Iterator[HyperPlane]:
        """
        Get the list of hyperplanes.
        Returns:
            list[HyperPlane]: List of hyperplanes.
        """
        for i in range(len(self)):
            yield HyperPlane(
                grid=self.grid,
                target=self.targets[i, :],
            )
