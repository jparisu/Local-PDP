"""
Core for data holding and efficient processing in Explainer module.
"""

from __future__ import annotations
from typing import Any, Callable
import logging
from dataclasses import dataclass

from faxai.explaining.configuration.DataCore import DataCore
from faxai.explaining.configuration.ExplainerConfiguration import ExplainerConfiguration
from faxai.explaining.ExplanationTechnique import ExplanationTechnique
from faxai.data.DataHolder import DataHolder

logger = logging.getLogger(__name__)

DEFAULT_CONFIGURATION_NAME = "__default__"

# Typing alias
ExplanationTechniqueCtor = Callable[..., ExplanationTechnique]


class ExplainerContainer:
    """
    Helper function to cache multiple explainers and their configurations.
    """

    configurations : dict[str, ExplainerConfiguration] = {}
    explainers : dict[str, dict[str, ExplanationTechnique]] = {}

    def add_configuration_and_technique(
            self,
            configuration_name: str,
            technique_name: str,
            configuration: ExplainerConfiguration,
            explainer: ExplanationTechnique
    ) -> None:
        """
        Add an explainer with its configuration to the container.

        Args:
            configuration_name (str): Name of the configuration.
            technique_name (str): Name of the explanation technique.
            configuration (ExplainerConfiguration): The explanation configuration instance.
            explainer (ExplanationTechnique): The explanation technique instance.
        """
        self.configurations[configuration_name] = configuration
        if configuration_name not in self.explainers:
            self.explainers[configuration_name] = {}
        self.explainers[configuration_name][technique_name] = explainer

    def add_technique(
            self,
            configuration_name: str,
            technique_name: str,
            explainer: ExplanationTechnique,
    ) -> None:
        """
        Add an explainer technique to the container without configuration.

        Args:
            technique_name (str): Name of the explanation technique.
            explainer (ExplanationTechnique): The explanation technique instance.
        """
        if configuration_name not in self.explainers:
            self.explainers[configuration_name] = {}
        self.explainers[configuration_name][technique_name] = explainer

    def get_technique(
            self,
            configuration_name: str,
            technique_name: str
    ) -> ExplanationTechnique | None:
        """
        Get an explainer by its configuration and technique names.

        Args:
            configuration_name (str): Name of the configuration.
            technique_name (str): Name of the explanation technique.

        Returns:
            ExplanationTechnique | None: The explanation technique instance if found, else None.
        """
        return self.explainers.get(configuration_name, {}).get(technique_name, None)

    def add_configuration(
            self,
            name: str,
            configuration: ExplainerConfiguration
    ) -> None:
        """
        Add an explanation configuration to the container.

        Args:
            name (str): Name of the configuration.
            configuration (ExplainerConfiguration): The explanation configuration instance.
        """
        self.configurations[name] = configuration

    def get_configuration(
            self,
            name: str
    ) -> ExplainerConfiguration | None:
        """
        Get an explanation configuration by its name.

        Args:
            name (str): Name of the configuration.

        Returns:
            ExplainerConfiguration | None: The explanation configuration instance if found, else None.
        """
        return self.configurations.get(name, None)


class ExplainerCore(ExplainerContainer):
    """
    Core class for data holding and efficient processing in Explainer module.

    This class holds the dataset (test or all) and the trained model to explain.
    It is used for efficient data processing and management within the Explainer module.
    """


    def __init__(
        self,
        datacore: DataCore
    ) -> None:
        """
        Initialize the ExplainerCore with dataframe and model.

        Args:
            model (Any): Trained model to explain.
            df (pd.DataFrame): DataFrame containing the feature data, not the target.

        Features are inferred from DataFrame columns.
        """
        # Store Data Core - main configuration with data and model
        self._datacore = datacore

        # Default parameters to create a configuration
        self._default_configuration_params : dict[str, Any] = {"use_default": True}


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


    def get_default_configuration(
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
            raise ValueError("Default configuration already has study features set, cannot override.")


        return ExplainerConfiguration(
            datacore=self.datacore(),
            study_features=features,
            **self._default_configuration_params
        )


    #####################
    # Getter

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
            technique: ExplanationTechniqueCtor | str,
            configuration: str | list[str] | ExplainerConfiguration,
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
        conf_name, configuration = self.__get_configuration(configuration)

        # Get actual technique
        tech_name, technique = self.__get_technique(technique)

        # Check if the explainer is already cached
        if conf_name != "" and tech_name != "":
            cached_explainer = self.get_technique(conf_name, tech_name)
            if cached_explainer is not None:
                logger.debug(f"Using cached explainer for technique '{tech_name}' and configuration '{conf_name}'.")
                return cached_explainer.explain(configuration, self)

        # Create the explainer
        explainer = technique(configuration)

        # Store the explanation in cache
        if conf_name != "" and tech_name != "":
            self.add_configuration_and_technique(
                conf_name,
                tech_name,
                configuration,
                explainer
            )

        # Generate explanations
        explanations = explainer.explain(configuration, self)

        return explanations


    def __get_configuration(
            self,
            configuration: str | list[str] | ExplainerConfiguration | None,
    ) -> tuple[str, ExplainerConfiguration]:
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
            return "", configuration

        # Case 2
        if configuration is None:
            return DEFAULT_CONFIGURATION_NAME, self.get_default_configuration()

        # Case 3
        if isinstance(configuration, str):
            config = self.get_configuration(configuration)
            if config is not None:
                return configuration, config

            # If not found, it may be a feature name
            if configuration in self.datacore().features():
                configuration = [configuration]

            else:
                raise ValueError(f"Configuration '{configuration}' not stored in core and do not refer to any feature.")

        # Case 4
        if isinstance(configuration, list):
            config = self.get_default_configuration(features=configuration)
            name = ','.join(configuration)
            self.add_configuration(name, config)
            return name, config

        # Error, non configuration found
        raise ValueError(f"Incorrect configuration.")



    def __get_technique(
            self,
            technique: ExplanationTechniqueCtor | str,
    ) -> tuple[str, ExplanationTechniqueCtor]:
        """
        To facilitate explainer technique retrieval, the method accepts multiple input types:

        1. ExplanationTechnique: already the technique instance.
        2. str: the name of a technique to create a new instance.

        In case 1, the technique will not be stored in the core.
        In case 2, a new technique will be created.
        """

        # Case 1
        # if isinstance(technique, ExplanationTechniqueCtor):
            # Use class name as technique name

        return technique.__class__.__name__, technique
        # TODO

    ####################
    #       PLOT       #
    ####################

    # TODO
