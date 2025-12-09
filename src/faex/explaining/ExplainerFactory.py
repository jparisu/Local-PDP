"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from faex.explaining.Explainer import Explainer, ExplainerPlot
from faex.utils.SingletonFactory import SingletonFactory

logger = logging.getLogger(__name__)


class ExplainerFactory:
    """
    Class to create explainers based on technique names.
    """

    def __init__(
            self,
            explainers: dict[str, Callable[..., Explainer]] | None = None,
            aliases: dict[str, str] | None = None) -> None:

        self._explainers : dict[str, Callable[..., Explainer]] = explainers if explainers is not None else {}
        self._aliases : dict[str, str] = aliases if aliases is not None else {}

        # Apply name convention to explainers and aliases
        self._explainers = {
            self.name_convention(name): explainer for name, explainer in self._explainers.items()
        }

        self._aliases = {
            self.name_convention(alias): self.name_convention(name) for alias, name in self._aliases.items()
        }


    def create_explainer(self, name: str, **kwargs: Any) -> Explainer:
        """
        Factory method to create an explainer object based on the technique name.

        Args:
            name (str): The name of the explainer name.

        Returns:
            Explainer: An instance of the requested explainer.
        """
        logger.debug("Creating explainer for technique: %s", name)
        explainer_class = self._get_explainer_class(name)
        return explainer_class(**kwargs)


    def _get_explainer_class(self, name: str) -> Callable[..., Explainer]:
        """
        Get the explainer class based on the technique name.

        Args:
            name (str): The name of the explainer technique.

        Returns:
            Callable[..., Explainer]: The explainer class.
        """
        real_name = self.name_convention(name)

        if real_name in self._explainers:
            return self._explainers[name]
        elif real_name in self._aliases:
            alias_name = self._aliases[real_name]
            return self._explainers[alias_name]

        raise ValueError(f"Explainer name '{name}' is not recognized.")

    def add_explainer(
        self,
        explainer: Callable[..., Explainer],
        aliases: list[str] | None = None,
    ) -> None:
        """
        Add an explainer if not already present.

        Args:
            explainer (Explainer ctor): The explainer to add.
            technique (str): The name of the explainer technique to add.
        """
        technique = explainer.name()

        technique = self.name_convention(technique)
        if technique not in self._explainers:
            self._explainers[technique] = explainer

        # Add aliases
        if aliases is not None:
            for alias in aliases:
                self.add_alias(alias, technique)


    def add_alias(self, alias: str, name: str) -> None:
        """
        Add an alias for an existing explainer technique.

        Args:
            alias (str): The alias name.
            name (str): The real explainer technique name.
        """
        real_name = self.name_convention(name)
        alias_name = self.name_convention(alias)

        if real_name not in self._explainers:
            raise ValueError(f"Cannot add alias '{alias}' for unknown explainer '{name}'.")

        self._aliases[alias_name] = real_name

    def get_available_explainers(self) -> list[str]:
        """
        Get the list of available explainer techniques.

        Returns:
            list[str]: List of available explainer technique names.
        """
        return list(self._explainers.keys())

    def get_available_plot_explainers(self) -> list[str]:
        """
        Get the list of available explainer techniques that support plotting.

        Returns:
            list[str]: List of available explainer technique names that support plotting.
        """
        plot_explainers = []
        for name, explainer_ctor in self._explainers.items():
            if issubclass(explainer_ctor, ExplainerPlot):
                plot_explainers.append(name)
        return plot_explainers

    def name_convention(self, technique: str) -> str:
        """
        Convert technique name to standard convention (lowercase with hyphens).

        Args:
            technique (str): The original technique name.

        Returns:
            str: The converted technique name.
        """
        cleaned_name = self.name_cleaning(technique)

        # Check if it is a known technique and return the standard name
        if cleaned_name in self._explainers:
            return cleaned_name

        if cleaned_name in self._aliases:
            return self._aliases[cleaned_name]

        return cleaned_name

    @classmethod
    def name_cleaning(cls, technique: str) -> str:
        """
        Clean technique name by removing spaces, hyphens, and underscores, and converting to lowercase.

        Args:
            technique (str): The original technique name.

        Returns:
            str: The cleaned technique name.
        """
        return technique.lower().replace("_", "").replace(" ", "").replace("-", "").strip()


    @staticmethod
    def register_explainer(
        explainer: Callable[..., Explainer],
        aliases: list[str] | None = None,
    ) -> None:
        """
        Register a new explainer globally.

        Args:
            explainer (Explainer ctor): The explainer to register.
            aliases (list[str], optional): List of aliases for the explainer technique.
        """

        # Get the global factory instance
        factory = GlobalExplainerFactory()

        # Check if the explainer is already registered
        real_name = factory.name_convention(explainer.name())

        if real_name in factory.get_available_explainers():
            raise ValueError(f"Explainer '{real_name}' is already registered.")

        logger.debug("Registering explainer '%s' with aliases: %s", real_name, aliases)

        # Add the explainer to the global factory
        factory.add_explainer(explainer, aliases=aliases)


# Singleton class of ExplainerFactory
GlobalExplainerFactory = SingletonFactory(ExplainerFactory)
GlobalExplainerFactoryInstance = GlobalExplainerFactory()
