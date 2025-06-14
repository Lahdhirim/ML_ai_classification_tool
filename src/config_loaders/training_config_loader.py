import json
from pydantic import BaseModel, Field
from typing import Dict, Optional

class FeatureSelectorConfig(BaseModel):
    """
    Configuration for feature selection.
    """
    features_path: str = Field(..., description="Path to features.")
    target_column: str = Field(..., description="Target column.")

class ModelConfig(BaseModel):
    """
    Configuration for an individual traditional ML model.
    Contains hyperparameters specific to various models such as Random Forest, XGBoost, Logistic Regression, or KNN.
    """
    enabled: bool = Field(..., description="Enable or disable the model.")
    n_estimators: Optional[int] = Field(None, description="Number of estimators.")
    max_depth: Optional[int] = Field(None, description="Max depth of the tree.")
    learning_rate: Optional[float] = Field(None, description="Learning rate for models like XGBoost.")
    n_neighbors: Optional[int] = Field(None, description="Number of neighbors for KNeighborsClassifier.")
    weights: Optional[str] = Field(None, description="Weights for KNeighborsClassifier.")
    criterion: Optional[str] = Field(None, description="Criterion for RandomForestClassifier.")
    C: Optional[float] = Field(None, description="Regularization parameter for LogisticRegression.")
    random_state: Optional[int] = Field(None, description="Random state for reproducibility.")

class MLPConfig(BaseModel):
    """
    Configuration for training a Multi-Layer Perceptron (MLP) model.
    """
    enabled: bool = Field(..., description="Enable or disable the MLP model.")
    hidden_layers_sizes: Optional[list] = Field(None, description="Sizes of hidden layers.")
    activation_function: Optional[str] = Field(None, description="Activation function for MLP hidden layers.")
    solver: Optional[str] = Field(None, description="Solver for MLP.")
    max_iter: Optional[int] = Field(None, description="Max iterations for MLP.")
    alpha: Optional[float] = Field(None, description="Regularization parameter for MLP.")
    random_state: Optional[int] = Field(None, description="Random state for MLP.")

class ModelsConfig(BaseModel):
    """
    Container for configurations of all machine learning models.

    Attributes:
        MLModels (Dict[str, ModelConfig]): Dictionary mapping model names to their configurations.
        MLP (MLPConfig): Configuration for the MLP model.
    """
    MLModels: Dict[str, ModelConfig]
    MLP: MLPConfig 

class TrainingConfig(BaseModel):
    """
    Master configuration for the model training pipeline.
    Includes data paths, feature selection, model definitions, and cross-validation setup.
    """
    train_processed_data_filename: str = Field(..., description="Path to the train processed data file")
    features_selector: FeatureSelectorConfig
    n_splits: int = Field(..., gt=0, description="Number of splits for cross-validation")
    models_params: ModelsConfig
    validation_raw_predictions_path: str = Field(..., description="Path to save validation raw predictions.")
    validation_kpis_path: str = Field(..., description="Filename to save validation KPIs.")
    trained_models_path: str = Field(..., description="Path to save trained models.")

def training_config_loader(config_path: str) -> TrainingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return TrainingConfig(**config)