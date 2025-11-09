"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from faex.data.DataHolder import DataHolder
from faex.data.DataPlotter import DataPlotter
from faex.explaining.DataCore import DataCore
from faex.explaining.Explainer import Explainer, ExplainerData, ExplainerPlot
from faex.explaining.ExplainerConfiguration import ExplainerConfiguration
from faex.explaining.ExplainerFactory import ExplainerFactory, GlobalExplainerFactory

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="Explainer")


class ExplainerContext:
    """
    Helper function to cache multiple explainers and their configurations.

    Attributes:
        datacore (DataCore): The data core containing the data to be explained.
        configuration (ExplainerConfiguration): The configuration for the explainers.
        explainers (dict[str, Explainer]): A dictionary of cached explainers.
        factory (ExplainerFactory): The factory to create explainers.
    """

    def __init__(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        explainers: dict[str, Explainer] | None = None,
        factory: ExplainerFactory | None = None,
    ):
        self.datacore = datacore
        self.configuration = configuration
        self.explainers = explainers if explainers is not None else {}
        self.factory = factory if factory is not None else GlobalExplainerFactory()

        # Apply name convention to existing explainers
        self.explainers = {
            ExplainerFactory.name_convention(name): explainer for name, explainer in self.explainers.items()
        }

    def __get_explainer_or_create(self, technique: str, forced_type: type[T]) -> T:
        """
        Get the explainer from the context, or create it using the factory if not present.

        Args:
            technique (str): The name of the explainer to use.
            forced_type (type[T]): The expected type of the explainer.

        Returns:
            Explainer: The requested explainer instance.
        """
        technique = ExplainerFactory.name_convention(technique)

        if technique not in self.explainers:
            # Try to create the explainer from the factory
            explainer = self.factory.create_explainer(technique)

            self.explainers[technique] = explainer

        explainer = self.explainers[technique]

        # If type is forced, check it
        if forced_type is not None and not isinstance(explainer, forced_type):
            raise TypeError(f"Explainer '{technique}' is not of type {forced_type.__name__}.")

        return explainer

    def explain(self, technique: str, **kwargs: Any) -> DataHolder:
        """
        Call the explain method of the specified explainer.
        If the technique is not yet in the context, look it up in Factory.

        Args:
            method (str): The name of the explainer to use.

        Returns:
            Any: The result of the explanation.
        """
        explainer = self.__get_explainer_or_create(technique, forced_type=ExplainerData)
        explainer.check_configuration(self.configuration, throw=True)
        return explainer.explain(context=self, **kwargs)

    def plot(self, technique: str, **kwargs: Any) -> DataPlotter:
        """
        Call the plot method of the specified explainer.
        If the technique is not yet in the context, look it up in Factory.

        Args:
            method (str): The name of the explainer to use.

        Returns:
            Any: The result of the plotting.
        """
        explainer = self.__get_explainer_or_create(technique, forced_type=ExplainerPlot)
        explainer.check_configuration(self.configuration, throw=True)
        return explainer.plot(context=self, **kwargs)
