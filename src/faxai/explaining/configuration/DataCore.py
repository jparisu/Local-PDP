"""
A explanation configuration class allows to set various parameters for generating explanations for
different techniques.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Any
import logging
import weakref

from faxai.utils.decorators import cache_method
from faxai.explaining.configuration.Configuration import Configuration

logger = logging.getLogger(__name__)


@dataclass
class DataCore(Configuration):
    """
    Core configuration class for explanation generation.

    This class holds the core parameters required for generating explanations.
    It includes the model and dataset.

    Attributes:
        model (Any): The machine learning model to be explained.
        df_X (pd.DataFrame): The input dataset used for generating explanations.
    """

    model: Any
    df_X: pd.DataFrame

    def __init__(self, model: Any, df_X: pd.DataFrame) -> None:
        """
        Initialize the CoreConfiguration with model and dataset.

        Args:
            model (Any): The machine learning model to be explained.
            df_X (pd.DataFrame): The input dataset used for generating explanations.
        """
        self.model = model
        self.df_X = df_X

        self.__cache_predictions_df: "weakref.WeakKeyDictionary[pd.DataFrame, np.ndarray]" = weakref.WeakKeyDictionary()


    def features(self) -> list[str]:
        """Get the feature names from the DataFrame."""
        return list(self.df_X.columns)

    def __len__(self) -> int:
        """Get the number of samples in the DataFrame."""
        return len(self.df_X)


    @cache_method
    def get_real_predictions(self) -> np.ndarray:
        """Get the real predictions from the model."""
        return self.model.predict(self.df_X)


    def predict(self, df_X: pd.DataFrame) -> np.ndarray:
        """
        Predict the target using the model for the given data.

        Args:
            df_X (pd.DataFrame): DataFrame containing the data to predict.

        Returns:
            np.ndarray: Predictions from the model.

        Note:
            It uses a weak reference to re-use predictions for data-frames already predicted.
        """

        # Check if predictions are cached
        # TODO: TypeError: unhashable type: 'DataFrame'
        # if df_X in self.__cache_predictions_df:
        #     logger.debug(f"Using cached predictions in {self} for the provided data {df_X}.")
        #     return self.__cache_predictions_df[df_X]

        # Predict
        predictions = self.model.predict(df_X)

        # Check predictions is a numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        # Store in cache
        # logger.debug(f"Storing predictions in cache in {self} for the provided data {df_X}.")
        # self.__cache_predictions_df[df_X] = predictions

        return predictions


    def valid(self, throw: bool = True) -> bool:
        """
        Validate the configuration parameters.

        Args:
            throw (bool): If True, raise an exception if the configuration is invalid.

        Returns:
            bool: True if the configuration is valid, False otherwise.
        """

        # Validate model is not None
        if self.model is None:
            if throw:
                raise ValueError("Model cannot be None.")
            return False

        # Validate DataFrame
        if self.df_X is None or not isinstance(self.df_X, pd.DataFrame):
            if throw:
                raise ValueError("df_X must be a valid pandas DataFrame.")
            return False

        # Check the DataFrame is a valid dataset for model.predict method
        try:
            _ = self.model.predict(self.df_X.head(1))
        except Exception as e:
            if throw:
                raise ValueError(f"The model and the dataset are not compatible: {e}") from e
            return False

        return True
