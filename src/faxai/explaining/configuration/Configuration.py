"""
A explanation configuration class allows to set various parameters for generating explanations for
different techniques.
"""

from __future__ import annotations
from abc import ABC, abstractmethod

class Configuration(ABC):
    """
    Generic configuration class for explanation generation.
    """

    @abstractmethod
    def valid(self, throw: bool = True) -> bool:
        """
        Validate the configuration parameters.

        Args:
            throw (bool): If True, raise an exception if the configuration is invalid.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        pass
