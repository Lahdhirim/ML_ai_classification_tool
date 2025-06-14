from src.config_loaders.training_config_loader import TrainingConfig
import pandas as pd
from colorama import Fore, Style
from sklearn.model_selection import StratifiedKFold
from src.modeling.features_selector import FeatureSelector
from sklearn.preprocessing import MinMaxScaler
from src.modeling.ml_models import MLModels
from src.modeling.mlp import MLPModel
from src.utils.schema import DatasetSchema, PipelinesDictSchema
from src.evaluators.ml_evaluator import MLEvaluator
import copy
from src.utils.model_saver import save_models
from src.utils.utils_toolbox import load_csv_data

class TrainingPipeline():
    
    """
    Pipeline for training and evaluating multiple machine learning models using cross-validation.

    This pipeline performs the following steps:
    1. Loads preprocessed training data from CSV.
    2. Splits the data into training and validation sets using Stratified K-Fold cross-validation.
    3. Applies feature selection and data scaling.
    4. Trains both traditional ML models and an MLP model.
    5. Evaluates model predictions and aggregates the results across folds.
    6. Saves the predictions, evaluation metrics, and trained pipelines for later use (e.g., testing or inference).

    Attributes:
        training_config (TrainingConfig): Configuration object containing paths, model settings, and CV strategy.
    """

    def __init__(self, training_config: TrainingConfig):
        self.training_config = training_config

    def run(self):
        print(Fore.YELLOW + "Running training pipeline..." + Style.RESET_ALL)

        # Load processed data
        df = load_csv_data(data_path = self.training_config.train_processed_data_filename)

        # Splitting data into train and validation sets using StratifiedKFold to ensure that the target variable is evenly distributed
        skf = StratifiedKFold(n_splits=self.training_config.n_splits, random_state=50, shuffle=True)

        # Initialize an empty DataFrame to store results of all folds
        results = pd.DataFrame()
        fold_index = 0

        # Initialize the dictionary of pipelines (useful for inference and testing): {fold_index: {feature_selector, scaler, ml_models, mlp_model}}
        pipelines_dict = {}

        for train_index, val_index in skf.split(df, df[self.training_config.features_selector.target_column]):
            current_pipeline = {}
            fold_index += 1
            print("-" * 40)
            print(f"{Fore.YELLOW}Fold {fold_index}{Style.RESET_ALL}")

            train, val = df.iloc[train_index], df.iloc[val_index]
            print(f"Train shape: {train.shape}, Validation shape: {val.shape}")

            # Feature selection
            features_selector = FeatureSelector(features_selector_config = self.training_config.features_selector)
            X_train, y_train = features_selector.transform(train)
            X_val, y_val = features_selector.transform(val)

            # Scaling
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Train models
            predictions = {}

            # Machine Learning Models
            ml_models = MLModels(config = self.training_config.models_params.MLModels)
            ml_models.train(X_train_scaled, y_train)
            predictions = ml_models.predict(X_val_scaled, predictions)

            # MLP Model
            mlp_model = MLPModel(config = self.training_config.models_params.MLP)
            mlp_model.train(X_train_scaled, y_train)
            predictions = mlp_model.predict(X_val_scaled, predictions)

            # Concatenate predictions
            predictions_df = pd.DataFrame(predictions)
            input_df = pd.DataFrame({DatasetSchema.FOLD: fold_index, 
                                     DatasetSchema.NAME: val[DatasetSchema.NAME],
                                     self.training_config.features_selector.target_column: y_val})
            results = results._append(pd.concat([input_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1), ignore_index=True)

            # Save current pipeline
            current_pipeline[PipelinesDictSchema.FEATURE_SELECTOR] = copy.deepcopy(features_selector)
            current_pipeline[PipelinesDictSchema.SCALER] = copy.deepcopy(scaler)
            if len(ml_models.models) > 0:
                current_pipeline[PipelinesDictSchema.ML_MODELS] = copy.deepcopy(ml_models)
            if mlp_model.model is not None:
                current_pipeline[PipelinesDictSchema.MLP_MODEL] = copy.deepcopy(mlp_model)
            pipelines_dict[fold_index] = current_pipeline

        # Save raw predictions
        results.to_csv(self.training_config.validation_raw_predictions_path, index=False)

        # Evaluate the predictions
        ml_evaluator = MLEvaluator(config = self.training_config, models = predictions.keys())
        ml_evaluator.evaluate(results)

        # Save pipelines in a pickle file
        save_models(pipelines_dict = pipelines_dict, output_file_name = self.training_config.trained_models_path)

        print(Fore.GREEN + "Training pipeline completed successfully." + Style.RESET_ALL)