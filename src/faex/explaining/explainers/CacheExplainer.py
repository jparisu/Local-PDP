"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from faex.data.DataHolder import DataHolder
from faex.explaining.DataCore import DataCore
from faex.explaining.Explainer import ExplainerData
from faex.explaining.ExplainerConfiguration import ExplainerConfiguration
from faex.utils.decorators import cache_method

# Avoid circular imports with TYPE_CHECKING
if TYPE_CHECKING:
    from faex.explaining.ExplainerContext import ExplainerContext

logger = logging.getLogger(__name__)


class CacheExplainerData(ExplainerData):
    """
    Abstract base class for explanation techniques that produce data outputs.
    """

    @cache_method
    def explain(self, context: ExplainerContext, **kwargs: Any) -> DataHolder:
        """
        Cache the explain method to avoid redundant computations.
        It also simplifies the interface by using ExplainerContext.
        """
        return self._explain(
            datacore=context.datacore,
            configuration=context.configuration,
            context=context,
        )

    @abstractmethod
    def _explain(
        self,
        datacore: DataCore,
        configuration: ExplainerConfiguration,
        context: ExplainerContext,
    ) -> DataHolder:
        """
        Generate explanations based on context, that holds configuration and data.
        """
        pass
