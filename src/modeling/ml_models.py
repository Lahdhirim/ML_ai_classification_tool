from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from colorama import Fore, Style
from src.config_loaders.training_config_loader import ModelsConfig
import pandas as pd
from typing import Dict
from sklearn.base import ClassifierMixin
import numpy as np

class MLModels:

    """
    Wrapper class to manage and train multiple traditional machine learning classifiers.

    This class supports scikit-learn models and XGBoost, initializing only those specified as `enabled`
    in the configuration.

    Supported models include:
    - KNeighborsClassifier
    - LogisticRegression
    - RandomForestClassifier
    - XGBoostClassifier

    Attributes:
        models (Dict[str, ClassifierMixin]): Dictionary of instantiated and configured models.

    Methods:
        train(X_train, y_train): Trains all enabled models on the given training data.
        predict(X, predictions): Makes predictions with all trained models and updates the given predictions dictionary.
    """

    def __init__(self, config: ModelsConfig):
        self.models = {}
        for model_name, model_config in config.items():
            if model_config.enabled:
                params = {k: v for k, v in model_config.dict().items() if k != "enabled" and v is not None}
                
                if model_name == "KNeighborsClassifier":
                    self.models["KNeighborsClassifier"] = KNeighborsClassifier(**params)
                elif model_name == "LogisticRegression":
                    self.models["LogisticRegression"] = LogisticRegression(**params)
                elif model_name == "RandomForest":
                    self.models["RandomForest"] = RandomForestClassifier(**params)
                elif model_name == "XGBoost":
                    self.models["XGBoost"] = xgb.XGBClassifier(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, ClassifierMixin]:
        for name, model in self.models.items():
            print(f"{Fore.BLUE}Training {name}{Style.RESET_ALL}")
            model.fit(X_train, y_train)
        return self.models
    
    def predict(self, X: pd.DataFrame, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for name, model in self.models.items():
            y_pred = model.predict(X)
            predictions[name] = y_pred 
        return predictions