"""
ClassifierModelHandler module for managing classifier models in Faex.
"""

class ClassifierModelHandler:
    """
    A handler class for managing classifier models.

    Feature Attribution commonly uses regression models.
    In order to use classification models, we need a dedicated handler.
    This handler selects one of the classes, and converts "predict" outputs
    to the probability of that class.

    Example usage:
        model = ...  # some classifier model
        handler = ClassifierModelHandler(model=model, class_to_handle="...")
    """

    def __init__(self, model, class_to_handle):
        """
        Initializes the ClassifierModelHandler with a model and the class to handle.

        Args:
            model: The classifier model to be managed.
            class_to_handle: The specific class for which probabilities will be computed.
        """
        self._model = model
        self._class_to_handle = class_to_handle

    def predict(self, X):
        """
        Predicts the probability of the specified class for the given input data.

        Args:
            X: Input data for which predictions are to be made.

        Returns:
            Probabilities of the specified class.
        """
        # Get the index of the class to handle
        class_index = list(self._model.classes_).index(self._class_to_handle)

        # Get the probabilities for all classes
        probabilities = self._model.predict_proba(X)

        # Return the probabilities for the specified class
        return probabilities[:, class_index]

    def get_model(self):
        """
        Returns the underlying classifier model.

        Returns:
            The classifier model.
        """
        return self._model

    def get_class_to_handle(self):
        """
        Returns the class that this handler is managing.

        Returns:
            The class to handle.
        """
        return self._class_to_handle
