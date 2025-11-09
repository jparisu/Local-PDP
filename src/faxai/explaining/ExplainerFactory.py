"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from faxai.explaining.Explainer import Explainer
from faxai.explaining.explainers.ICE import ICE, ICE_Scatter
from faxai.explaining.explainers.PDP import PDP
from faxai.explaining.explainers.M_ICE import M_ICE
from faxai.explaining.explainers.M_PDP import M_PDP
from faxai.explaining.explainers.data import Histogram, RealPrediction
from faxai.explaining.explainers.kernel import KernelValues, KernelNormalizer
from faxai.explaining.explainers.L_ICE import L_ICE
from faxai.explaining.explainers.L_PDP import L_PDP


from faxai.utils.SingletonFactory import SingletonFactory

logger = logging.getLogger(__name__)


class ExplainerFactory:
    """
    Class to create explainers based on technique names.
    """

    ExplainersAvailable = {
        ICE,
        PDP,
        M_ICE,
        M_PDP,
        Histogram,
        RealPrediction,
        KernelValues,
        KernelNormalizer,
        L_ICE,
        L_PDP,
        ICE_Scatter,
    }


    def __init__(self, explainers: dict[str, Callable[..., Explainer]] | None = None) -> None:
        if explainers is None:
            self._explainers = {}
            for explainer in ExplainerFactory.ExplainersAvailable:
                name = self.name_convention(explainer.name())
                self._explainers[name] = explainer
        else:
            self._explainers = explainers


    def create_explainer(self, technique: str, **kwargs: Any) -> Explainer:
        """
        Factory method to create an explainer based on the technique name.

        Args:
            technique (str): The name of the explainer technique.

        Returns:
            Explainer: An instance of the requested explainer.
        """
        technique = self.name_convention(technique)

        if technique in self._explainers:
            explainer_class = self._explainers[technique]
            return explainer_class(**kwargs)
        else:
            raise ValueError(f"Explainer technique '{technique}' is not recognized.")


    def add_explainer(
        self,
        explainer: Callable[..., Explainer],
        technique: str | None = None,
    ) -> None:
        """
        Add an explainer if not already present.

        Args:
            explainer (Explainer ctor): The explainer to add.
            technique (str): The name of the explainer technique to add.
        """
        if technique is None:
            technique = explainer.name()

        technique = self.name_convention(technique)
        if technique not in self._explainers:
            self._explainers[technique] = explainer


    def get_available_explainers(self) -> list[str]:
        """
        Get the list of available explainer techniques.

        Returns:
            list[str]: List of available explainer technique names.
        """
        return list(self._explainers.keys())


    @classmethod
    def name_convention(cls, technique: str) -> str:
        """
        Convert technique name to standard convention (lowercase with hyphens).

        Args:
            technique (str): The original technique name.

        Returns:
            str: The converted technique name.
        """
        return technique.lower().replace("_", "").replace(" ", "").replace("-", "").strip()


# Singleton instance of ExplainerFactory
GlobalExplainerFactory = SingletonFactory(ExplainerFactory)
