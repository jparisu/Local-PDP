"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

from faxai.utils.decorators import cache_method
from faxai.explaining.configuration.ExplainerConfiguration import ExplainerConfiguration
from faxai.data.DataHolder import DataHolder

logger = logging.getLogger(__name__)



class ExplanationTechnique(ABC):
    """
    This is an abstract base class for different explanation techniques.
    It defines the common interface and structure for all explanation classes.
    """

    #######################
    # Static Methods

    @classmethod
    @abstractmethod
    def check_configuration(
            cls,
            configuration: ExplainerConfiguration,
            throw: bool = True
    ) -> bool:
        """
        Check if the provided configuration is valid for this explanation technique.

        Args:
            configuration (ExplainerConfiguration): The configuration to check.

        Returns:
            List[ExplainerConfiguration]: A list of valid configurations.
        """
        pass


    # @classmethod
    # @abstractmethod
    # def plot_methods(
    #     cls,
    #     explanation: DataHolder,
    #     params: Optional[Dict[str, Any]] = None
    # ) -> Any:
    #     """
    #     Plot the explanation results.

    #     Args:
    #         explanation (DataHolder): The explanation data to plot.
    #         params (Optional[Dict[str, Any]]): Additional parameters for plotting.

    #     Returns:
    #         Any: The plotting object or result.
    #     """
    #     pass


    #####################
    # Object Methods

    def __init__(
        self,
        configuration: ExplainerConfiguration,
    ):
        """
        Initialize the ExplanationTechnique with a configuration.
        """

        # Check the configuration is valid
        self.check_configuration(configuration, throw=True)

        self.configuration = configuration

        self._cache : DataHolder | None = None



    def explain(
        self,
        configuration: ExplainerConfiguration,
        core: ExplainerCore | None = None,
    ) -> DataHolder:
        """
        Generate explanations based on the provided configuration.
        It can use core if provided for efficient data handling.

        It stores the explanation in cache for future calls.

        Args:
            configuration (ExplainerConfiguration): The configuration for the explanation.
            core (ExplainerCore): The core data and model information.

        Returns:
            ExplanationTechnique: An instance of the explanation technique with computed explanations.
        """
        # Check if cache
        if self._cache is not None:
            logger.debug(f"Using cached explanation in {self}.")
            return self._cache

        # Generate explanation
        explanation = self._explain(configuration, core)
        logger.debug(f"Storing explanation in cache in {self}.")
        self._cache = explanation

        return explanation

    @abstractmethod
    def _explain(
        self,
        configuration: ExplainerConfiguration,
        core: ExplainerCore | None = None,
    ) -> DataHolder:
        """
        Internal method to generate explanations based on the provided configuration.
        It can use core if provided for efficient data handling.

        Args:
            configuration (ExplainerConfiguration): The configuration for the explanation.
            core (ExplainerCore): The core data and model information.

        Returns:
            ExplanationTechnique: An instance of the explanation technique with computed explanations.
        """
        pass
