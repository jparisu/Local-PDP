"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations
from typing import Any, Callable
import logging
from dataclasses import dataclass

from faxai.explaining.Explainer import Explainer
from faxai.explaining.explainers.ICE import ICE
from faxai.explaining.explainers.PDP import PDP
from faxai.utils.SingletonFactory import SingletonFactory


logger = logging.getLogger(__name__)


class ExplainerFactory:
    """
    Class to create explainers based on technique names.
    """

    ExplainersAvailable = {
        "ice": ICE,
        "pdp": PDP,
    }

    def __init__(self, explainers: dict[str, Callable[..., Explainer]] | None = None) -> None:

        if explainers is None:
            explainers = ExplainerFactory.ExplainersAvailable

        self._explainers = explainers


    def create_explainer(
        self,
        technique: str,
        **kwargs: Any
    ) -> Explainer:
        """
        Factory method to create an explainer based on the technique name.

        Args:
            technique (str): The name of the explainer technique.

        Returns:
            Explainer: An instance of the requested explainer.
        """
        technique = technique.lower()
        if technique in ExplainerFactory.ExplainersAvailable:
            explainer_class = ExplainerFactory.ExplainersAvailable[technique]
            return explainer_class(**kwargs)
        else:
            raise ValueError(f"Explainer technique '{technique}' is not recognized.")


    def add_explainer(
        self,
        explainer: Callable[..., Explainer],
        technique: str,
    ) -> None:
        """
        Add an explainer if not already present.

        Args:
            explainer (Explainer ctor): The explainer to add.
            technique (str): The name of the explainer technique to add.
        """
        technique = technique.lower()
        if technique not in ExplainerFactory.ExplainersAvailable:
            ExplainerFactory.ExplainersAvailable[technique] = explainer


    def get_available_explainers(self) -> list[str]:
        """
        Get the list of available explainer techniques.

        Returns:
            list[str]: List of available explainer technique names.
        """
        return list(ExplainerFactory.ExplainersAvailable.keys())


# Singleton instance of ExplainerFactory
GlobalExplainerFactory = SingletonFactory(ExplainerFactory)
