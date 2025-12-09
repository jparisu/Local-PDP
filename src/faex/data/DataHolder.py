from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from faex.mathing.distribution.Distribution import Distribution
from faex.mathing.distribution.sampling_distributions import DeltaDistribution, DeltaWeightedDistribution

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

    """
    TODO:
    - len, dim, shape, get_item
    """


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

    def max(self) -> float:
        """
        Get the maximum value of the hyperplane target.
        Returns:
            float: Maximum value of the hyperplane target.
        """
        return float(np.max(self.target))

    def min(self) -> float:
        """
        Get the minimum value of the hyperplane target.
        Returns:
            float: Minimum value of the hyperplane target.
        """
        return float(np.min(self.target))


    def narrow(self, max_value: float = 1.0, min_value: float = 0.0) -> HyperPlane:
        """
        Normalize the hyperplane target values relative to their maximum.

        By default, scales so that max(target) -> max_value and others are
        proportional to that (i.e., roughly between min_value and max_value,
        without forcing the minimum to be exactly min_value).

        Example:
            target = [100, 90, 90, 80]
            narrow() -> [1.0, 0.9, 0.9, 0.8]

        Args:
            max_value (float): The value that the maximum target will map to.
            min_value (float | None): Baseline after scaling. If None, uses 0.0.

        Returns:
            HyperPlane: A new HyperPlane with narrowed target values.
        """

        if min_value > max_value:
            raise ValueError(
                f"min_value ({min_value}) cannot be greater than max_value ({max_value})."
            )

        current_max = self.max()

        # Handle edge case: all values are zero / same
        if current_max == 0:
            # everything is 0 → just fill with min_value (or 0 if you prefer)
            narrowed_target = np.full_like(self.target, fill_value=min_value, dtype=float)
        else:
            # Step 1: scale so that max becomes 1
            scaled = self.target / current_max

            # Step 2: stretch into [min_value, max_value] without pinning min to min_value
            # max(scaled) = 1 -> maps to max_value
            # others are proportional: min(scaled) stays > min_value unless it's 0
            range_ = max_value - min_value
            narrowed_target = scaled * range_ + min_value

        return HyperPlane(
            grid=self.grid,
            target=narrowed_target,
        )

@dataclass
class HyperPlanes(DataHolder):
    """
    A data class representing a N dimensional grid and M target dimensions with a value for each point in the grid.
    Every hyperplane inside targets has the same grid.

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
        Get the iterator of internal hyperplanes.
        Returns:
            Iterator[HyperPlane]: List of hyperplanes.
        """
        for i in range(len(self)):
            yield HyperPlane(
                grid=self.grid,
                target=self.targets[i, :],
            )

    def max(self) -> float:
        """
        Get the maximum value of the hyperplanes targets.
        Returns:
            float: Maximum value of the hyperplanes targets.
        """
        return float(np.max(self.targets))

    def min(self) -> float:
        """
        Get the minimum value of the hyperplanes targets.
        Returns:
            float: Minimum value of the hyperplanes targets.
        """
        return float(np.min(self.targets))


    def to_distributions(self) -> DistributionCollection:
        """
        Convert the hyperplanes to a DistributionCollection where each point in the grid has a distribution
        formed by the values of the hyperplanes at that point.

        Returns:
            DistributionCollection: The resulting DistributionCollection.
        """
        targets = self.targets
        if targets.ndim < 2:
            raise ValueError("targets must have at least 2 dimensions: (M, A1, ..., AN)")

        M = targets.shape[0]
        grid_shape = targets.shape[1:]          # (A1, A2, ..., AN)
        num_points = int(np.prod(grid_shape))   # total number of grid points

        # Flatten spatial dimensions: (M, A1, ..., AN) -> (num_points, M)
        flat_targets = targets.reshape(M, num_points).T  # shape: (num_points, M)

        # Build one Distribution per grid point
        flat_distributions = np.array(
            [DeltaDistribution(samples=row) for row in flat_targets],
            dtype=object,                        # array of Python objects (Distribution)
        )

        # Reshape back to grid shape: (A1, A2, ..., AN)
        distributions = flat_distributions.reshape(grid_shape)

        return DistributionCollection(
            grid=self.grid,
            distributions=distributions,
        )


@dataclass
class WeightedHyperPlane(HyperPlane):
    """
    A data class representing a N dimensional grid and a weighted target dimension.
    The target has a value and a weight for each point in the grid.

    Attributes:
        grid (Grid): N dimensional base data with shape A1 x A2 x ... x AN.
        target (np.ndarray): matrix with shape (A1, A2, ..., AN) representing the target values for each point in the grid.
        weights (np.ndarray): matrix with shape (A1, A2, ..., AN) representing the weights for each point in the grid.
    """

    weights: np.ndarray


@dataclass
class WeightedHyperPlanes(HyperPlanes):
    """
    A data class representing a N dimensional grid and M weighted target dimensions.
    Every hyperplane inside targets has the same grid.
    Each target has a value and a weight for each point in the grid.

    Attributes:
        grid (Grid): N dimensional base data with shape A1 x A2 x ... x AN.
        targets (np.ndarray): matrix with shape (M, A1, A2, ..., AN) representing the target values for each point in the grid.
        weights (np.ndarray): matrix with shape (M, A1, A2, ..., AN) representing the weights for each point in the grid.
    """

    weights: np.ndarray

    def it_weighted_hyperplanes(self) -> Iterator[WeightedHyperPlane]:
        """
        Get the iterator of internal weighted hyperplanes.
        Returns:
            Iterator[WeightedHyperPlane]: List of weighted hyperplanes.
        """
        for i in range(len(self)):
            yield WeightedHyperPlane(
                grid=self.grid,
                target=self.targets[i, :],
                weights=self.weights[i, :],
            )

    def to_distributions(self, max_weight: float = None) -> DistributionCollection:
        """
        Convert the weighted hyperplanes to a DistributionCollection where each point in the grid has a distribution
        formed by the values of the hyperplanes at that point, weighted by their respective weights.

        Returns:
            DistributionCollection: The resulting DistributionCollection.
        """
        targets = self.targets
        if targets.ndim < 2:
            raise ValueError("targets must have at least 2 dimensions: (M, A1, ..., AN)")

        M = targets.shape[0]
        grid_shape = targets.shape[1:]          # (A1, A2, ..., AN)
        num_points = int(np.prod(grid_shape))   # total number of grid points

        # Flatten spatial dimensions: (M, A1, ..., AN) -> (num_points, M)
        flat_targets = targets.reshape(M, num_points).T  # shape: (num_points, M)
        flat_weights = self.weights.reshape(M, num_points).T  # shape: (num_points, M)

        # Build one Distribution per grid point
        flat_distributions = np.array(
            [DeltaWeightedDistribution(samples=flat_targets[i], weights=flat_weights[i], max_weight=max_weight) for i in range(num_points)],
            dtype=object,                        # array of Python objects (Distribution)
        )

        # Reshape back to grid shape: (A1, A2, ..., AN)
        distributions = flat_distributions.reshape(grid_shape)

        return DistributionCollection(
            grid=self.grid,
            distributions=distributions,
        )


class DataHolderCollection(DataHolder):
    """
    A class representing a collection of data holders.

    Attributes:
        data_holders (list[DataHolder]): List of data holders.
    """

    def __init__(self, data_holders: list[DataHolder] | None = None) -> None:
        if data_holders is None:
            data_holders = []

        self.data_holders = data_holders

    def add(self, data_holder: DataHolder) -> None:
        """
        Add a data holder to the collection.

        Args:
            data_holder (DataHolder): The data holder to add.
        """
        self.data_holders.append(data_holder)

    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the data collection.
        """
        return (len(self.data_holders),)

    def __len__(self) -> int:
        """
        Get the number of data holders inside.
        """
        return len(self.data_holders)

    def __iter__(self) -> Iterator[DataHolder]:
        """
        Get the iterator of data holders.
        Returns:
            Iterator[DataHolder]: Iterator of data holders.
        """
        return iter(self.data_holders)



@dataclass
class DistributionCollection(DataHolder):
    """
    A data class representing a N dimensional grid where each point has an associated distribution.

    Attributes:
        grid (Grid): N dimensional base data with shape A1 x A2 x ... x AN.
        distributions (list[Distribution]): List of distributions for each point in the grid.
    """

    grid: Grid
    distributions: np.ndarray[Distribution]

    def shape(self) -> tuple[int, ...]:
        """
        Get the shape of the distribution collection.
        Returns:
            tuple[int, ...]: Shape of the distribution collection.
        """
        return self.grid.shape()
