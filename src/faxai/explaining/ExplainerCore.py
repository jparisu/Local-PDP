"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations
from typing import Any, Callable
import logging
from dataclasses import dataclass
import pandas as pd

from faxai.explaining.DataCore import DataCore
from faxai.explaining.ExplainerContext import ExplainerContext
from faxai.explaining.ExplainerConfiguration import ExplainerConfiguration
from faxai.data.DataHolder import DataHolder
from faxai.data.DataPlotter import DataPlotter

logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION_NAME = "__default__"

class ExplainerCore:
    """
    Core class for data holding and efficient processing in Explainer module.

    This class holds the dataset (test or all) and the trained model to explain.
    It is used for efficient data processing and management within the Explainer module.
    """


    def __init__(
        self,
        dataframe_X: pd.DataFrame | None = None,
        model: Any | None = None,
        datacore: DataCore | None = None,
        configurations: dict[str, ExplainerConfiguration] | None = None,
    ) -> None:
        """
        Initialize the ExplainerCore with dataframe and model.

        Args:
            model (Any): Trained model to explain.
            df (pd.DataFrame): DataFrame containing the feature data, not the target.

        Features are inferred from DataFrame columns.
        """

        # Create DataCore if not provided
        if datacore is None:
            if dataframe_X is None or model is None:
                raise ValueError("Either datacore or both dataframe_X and model must be provided.")

            datacore = DataCore(
                df_X=dataframe_X,
                model=model,
            )

        # Store Data Core - main configuration with data and model
        self._datacore = datacore

        # Create contexts
        self._contexts = {}

        if configurations is not None:
            for name, config in configurations.items():
                self._contexts[name] = ExplainerContext(
                    datacore=self._datacore,
                    configuration=config,
                )

        # Set default configuration params
        self._default_configuration_params: dict[str, Any] = {}


    #####################
    # Configuration

    def set_default_configuration_params(
            self,
            params: dict[str, Any]
    ) -> None:
        """
        Set the default explanation configuration.

        Args:
            configuration (ExplainerConfiguration): The explanation configuration instance.
        """
        self._default_configuration_params = params


    def __get_default_configuration(
            self,
            features: list[str] | None = None,
    ) -> ExplainerConfiguration:
        """
        Get the default explanation configuration.

        Returns:
            ExplainerConfiguration: The default explanation configuration instance.
        """

        # Remove feature_study from params if exists
        params = self._default_configuration_params.copy()
        features_params = params.pop('study_features', None)

        if not features:
            features = features_params

        elif features and features_params and features != features_params:
            raise ValueError(f"Default configuration already has study features set {features_params}, cannot override.")

        return ExplainerConfiguration(
            datacore=self.datacore(),
            study_features=features,
            **self._default_configuration_params
        )


    def __new_configuration_id(self) -> str:
        """
        Generate a new unique configuration ID.

        Returns:
            str: The new configuration ID.
        """
        return "__config__" + str(len(self._contexts) + 1)


    def add_configuration(
            self,
            name: str,
            configuration: ExplainerConfiguration | None = None,
            override: bool = True,
    ) -> None:
        """
        Add an explanation configuration to the core.

        Args:
            configuration (ExplainerConfiguration): The explanation configuration instance.
        """
        if name in self._contexts and not override:
            raise ValueError(f"Configuration '{name}' already exists in core.")

        if configuration is None:
            configuration = self.__get_default_configuration()

        # Check configuration is valid
        configuration.check(throw=True)

        self._contexts[name] = ExplainerContext(
            datacore=self._datacore,
            configuration=configuration,
        )


    #####################
    # Getters

    def datacore(self) -> DataCore:
        """
        Get the DataCore instance.

        Returns:
            DataCore: The DataCore instance containing the dataset and model.
        """
        return self._datacore


    #####################
    #      EXPLAIN      #
    #####################

    def explain(
            self,
            technique: str,
            configuration: str | list[str] | ExplainerConfiguration | None,
            **kwargs,
    ) -> DataHolder:

        """
        Generate explanations using the specified technique and configuration.

        Args:
            technique (ExplanationTechnique): The explanation technique to use.
            configuration_name (str): The name of the configuration to use.

        Returns:
            Any: The generated explanations.
        """
        # Get actual configuration
        configuration_name = self.__resolve_configuration_name(configuration)

        # Generate explanations
        return self._contexts[configuration_name].explain(technique, **kwargs)


    def plot(
            self,
            technique: str,
            configuration: str | list[str] | ExplainerConfiguration | None,
            **kwargs,
    ) -> DataPlotter:

        """
        Generate explanations using the specified technique and configuration.

        Args:
            technique (ExplanationTechnique): The explanation technique to use.
            configuration_name (str): The name of the configuration to use.

        Returns:
            Any: The generated explanations.
        """
        # Get actual configuration
        configuration_name = self.__resolve_configuration_name(configuration)

        # Generate explanations
        return self._contexts[configuration_name].plot(technique, **kwargs)


    def __resolve_configuration_name(
            self,
            configuration: str | list[str] | ExplainerConfiguration | None,
    ) -> str:
        """
        To facilitate explainer configuration retrieval, the method accepts multiple input types:

        1. ExplainerConfiguration: already the configuration instance.
        2. None: to get the default configuration.
        3. str: the name of a single configuration stored in the core.
        4. str | list[str] : name of a feature or list of features to create a default configuration for them.

        In case 1, the configuration will not be stored in the core.
        In case 4, a new configuration will be created and stored in the core with the name ','.join(features)
        Case 2 may fail if default configuration has no feature associated, what must be set by user.
        """

        # Case 1
        if isinstance(configuration, ExplainerConfiguration):
            id = self.__new_configuration_id()
            self.add_configuration(id, configuration)
            return id

        # Case 2
        if configuration is None:
            return DEFAULT_CONFIGURATION_NAME

        # Case 3
        if isinstance(configuration, str):
            if configuration in self._contexts:
                return configuration

            # If not found, it may be a feature name
            if configuration in self.datacore().features():
                configuration = [configuration]

            else:
                raise ValueError(f"Configuration '{configuration}' not stored in core and do not refer to any feature.")

        # Case 4
        if isinstance(configuration, list):

            # Feature list provided
            name = ','.join(configuration)
            # Check if the configuration already exists
            if name in self._contexts:
                return name

            # Create default configuration
            config = self.__get_default_configuration(features=configuration)
            self.add_configuration(name, config)
            return name

        # Error, non configuration found
        raise ValueError(
            f"Incorrect configuration. Set the name of a stored configuration, or the name of a feature or features"
            +"to study with default configuration.")

    ####################
    #       PLOT       #
    ####################

    # TODO
