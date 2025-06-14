from sklearn.neural_network import MLPClassifier
import numpy as np
from colorama import Fore, Style
import pandas as pd
from typing import Dict
from src.config_loaders.training_config_loader import MLPConfig
from sklearn.base import ClassifierMixin
from src.utils.schema import PredictionSchema

class MLPModel:

    """
    Wrapper class for configuring, training, and predicting with a Multi-Layer Perceptron (MLP) model.

    This class uses `sklearn.neural_network.MLPClassifier` and is conditionally enabled via the configuration.

    Attributes:
        model (MLPClassifier or None): Instantiated MLP model if enabled, else None.

    Methods:
        train(X_train, y_train): Trains the MLP model on provided training data.
        predict(X, predictions): Makes predictions using the trained MLP and updates the predictions dictionary.
    """

    def __init__(self, config: MLPConfig):
        if config.enabled:
            params = {
                'hidden_layer_sizes': config.hidden_layers_sizes or (100, 100),
                'activation': config.activation_function or 'logistic',
                'solver': config.solver or 'adam',
                'max_iter': config.max_iter or 10000,
                'random_state': 42
            }
            self.model = MLPClassifier(**params)
        else:
            self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
        if self.model is not None:
            print(f"{Fore.BLUE}Training MLP Model{Style.RESET_ALL}")
            self.model.fit(X_train, y_train)
            return self.model

    def predict(self, X: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.model is not None:
            y_pred = self.model.predict(X)
            predictions[PredictionSchema.MLP] = y_pred
        return predictions