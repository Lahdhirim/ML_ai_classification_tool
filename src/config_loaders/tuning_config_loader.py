import json
from pydantic import BaseModel, Field
from typing import Dict, Union, List, Literal
from src.config_loaders.training_config_loader import FeatureSelectorConfig

class IntParam(BaseModel):
    type: Literal["int"]
    low: int
    high: int

class FloatParam(BaseModel):
    type: Literal["float"]
    low: float
    high: float

class CategoricalParam(BaseModel):
    type: Literal["categorical"]
    choices: List[str]

HyperParam = Union[IntParam, FloatParam, CategoricalParam]
"""
    Union of different hyperparameter types used in Optuna tuning.

    This type alias allows a hyperparameter to be of one of the following types:
        - IntParam: An integer hyperparameter with specified low and high bounds.
        - FloatParam: A floating-point hyperparameter with specified low and high bounds.
        - CategoricalParam: A categorical hyperparameter with a list of valid choices.

    Each hyperparameter will have a specific type, and can be used to define the range or categories 
    that Optuna will optimize during the tuning process.
"""

class OptunaConfig(BaseModel):
    """
    Configuration for the Optuna hyperparameter optimization process.

    This class defines the parameters for conducting hyperparameter tuning using the Optuna framework. 
    It specifies the number of trials, cross-validation folds, the scoring metric to optimize, and the 
    models and their hyperparameters to tune.
    """
    n_trials: int = Field(..., gt=0, description="Number of Optuna trials")
    cv: int = Field(..., gt=1, description="Number of Cross-Validation folds")
    scoring: str = Field("recall", description="Scoring metric for Optuna")
    direction: str = Field("maximize", description="Direction of optimization")
    models: Dict[str, Dict[str, HyperParam]]

class TuningConfig(BaseModel):
    """
    Configuration for hyperparameter tuning using Optuna.

    This class encapsulates all the necessary configurations for tuning models' hyperparameters
    using the Optuna optimization framework. It includes settings for cross-validation, scoring metrics,
    and the specific models and their hyperparameters to be tuned.
    """
    train_processed_data_filename: str = Field(..., description="Path to the train processed data file")
    features_selector: FeatureSelectorConfig
    best_hyperparams_filename: str = Field(..., description="Path to save the best hyperparameters")
    optuna: OptunaConfig

def tuning_config_loader(config_path: str) -> TuningConfig:
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return TuningConfig(**config_dict)
