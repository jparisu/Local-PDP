"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from faxai.data.DataHolder import DataHolder
from faxai.data.DataPlotter import DataPlotter
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faxai.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class Explainer(ABC):
    """
    This is an abstract base class for different explanation techniques.
    It defines the common interface and structure for all explanation classes.
    """

    #######################
    # Static Methods

    @classmethod
    def check_configuration(cls, configuration: ExplainerConfiguration, throw: bool = True) -> bool:
        """
        Check if the provided configuration is valid for this explanation technique.

        By default, check datacore presence.
        Args:
            configuration (ExplainerConfiguration): The configuration to check.

        Returns:
            List[ExplainerConfiguration]: A list of valid configurations.
        """
        return configuration.check(throw=throw)


    @classmethod
    def name(cls) -> str:
        """
        Get the name of the explanation technique.

        Returns:
            str: The name of the explanation technique.
        """
        return cls.__name__.lower()


class ExplainerData(Explainer):
    """
    Abstract base class for explanation techniques that produce data outputs.
    """

    @abstractmethod
    def explain(self, context: ExplainerContext, **kwargs: Any) -> DataHolder:
        """
        Generate explanations based on context, that holds configuration and data.
        """
        pass


class ExplainerPlot(Explainer):
    """
    Abstract base class for explanation techniques that involve plotting.
    """

    @abstractmethod
    def plot(self, context: ExplainerContext, **kwargs: Any) -> DataPlotter:
        """
        Generate plots based on the explanations in context.
        """
        pass
