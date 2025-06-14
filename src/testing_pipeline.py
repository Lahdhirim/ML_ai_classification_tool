from src.config_loaders.testing_config_loader import TestingConfig
from src.utils.utils_toolbox import load_csv_data
from colorama import Fore, Style
from src.utils.model_loader import load_models
from src.utils.schema import PipelinesDictSchema, DatasetSchema, PredictionSchema, EvaluatorSchema
import pandas as pd
from typing import List
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class TestingPipeline:
    
    """
    A pipeline for testing the performance of trained models on processed data.

    This class loads previously processed test data and trained model pipelines, performs predictions, aggregates the results,
    and evaluates the models using common classification metrics. It then saves the evaluation metrics to an Excel file.

    Attributes:
        config (TestingConfig): Configuration object containing paths and parameters for the testing process.

    Methods:
        predictions_aggregation(data, models, n_folds): Aggregates predictions across multiple folds or models.
        evaluate_predictions(data, models): Evaluates the model predictions using accuracy, recall, precision, and F1 score.
        run(): Executes the entire testing pipeline, including loading data, making predictions, and evaluating results.
    """
    
    def __init__(self, config: TestingConfig):
        super().__init__(config)

    def predictions_aggregation(self, data: pd.DataFrame, models: List[str], n_folds: int) -> None:
        data_copy = data.copy()
        if n_folds > 1: # Calculate the mean of the predictions across all folds
            data_copy = data_copy.groupby([DatasetSchema.ID, DatasetSchema.TARGET])[models].mean().reset_index()     
        else:
            data_copy = data_copy.drop(DatasetSchema.FOLD, axis=1)
        if len(models) > 1: # If there are multiple models, calculate the aggregated prediction
            data_copy[PredictionSchema.PREDICTION_AGGREGATED] = (data_copy[models].mean(axis=1) >= self.config.probability_threshold)
        data_copy[models + [PredictionSchema.PREDICTION_AGGREGATED]] = data_copy[models + [PredictionSchema.PREDICTION_AGGREGATED]].astype(int)
        return data_copy

    def evaluate_predictions(self, data: pd.DataFrame, models: List[str]) -> None:
        y_true = data[DatasetSchema.TARGET]
        results = {}

        if PredictionSchema.PREDICTION_AGGREGATED in data.columns:
            models.append(PredictionSchema.PREDICTION_AGGREGATED)
            
        for model in models:
            y_pred = data[model]
            accuracy = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            results[model] = {
                EvaluatorSchema.ACCURACY: accuracy,
                EvaluatorSchema.RECALL: recall,
                EvaluatorSchema.PRECISION: precision,
                EvaluatorSchema.F1: f1
            }

        # Save results to Excel file
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.name = PredictionSchema.MODEL
        with pd.ExcelWriter(self.config.test_kpis_path) as writer:
            results_df.to_excel(writer, sheet_name="Evaluation")
            print(f"KPIs Results saved to {self.config.test_kpis_path}")

    
    def run(self):
        print(Fore.YELLOW + "Running testing pipeline..." + Style.RESET_ALL)

        # Load processed data
        df = load_csv_data(data_path=self.config.test_processed_data_filename)

        # Load trained pipelines
        pipelines_dict = load_models(saved_models_path = self.config.saved_models_path)
        results = pd.DataFrame()

        for pipeline_index, steps in pipelines_dict.items():
            
            # Feature selection
            features_selector = steps[PipelinesDictSchema.FEATURE_SELECTOR]
            X_test, y_test = features_selector.transform(df)

            # Scaling
            scaler = steps[PipelinesDictSchema.SCALER]
            X_test_scaled = scaler.transform(X_test)

            # Model predictions
            predictions = {}
            if PipelinesDictSchema.ML_MODELS in steps:
                ml_models = steps[PipelinesDictSchema.ML_MODELS]
                predictions = ml_models.predict(X_test_scaled, predictions)
            
            if PipelinesDictSchema.MLP_MODEL in steps:
                mlp_model = steps[PipelinesDictSchema.MLP_MODEL]
                predictions = mlp_model.predict(X_test_scaled, predictions)
            
            # Concatenate predictions
            predictions_df = pd.DataFrame(predictions)
            input_df = pd.DataFrame({DatasetSchema.FOLD: pipeline_index, 
                                     DatasetSchema.ID: df[DatasetSchema.ID],
                                     DatasetSchema.TARGET: y_test})
            results = results._append(pd.concat([input_df.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1), ignore_index=True)

        # Save raw predictions
        results = self.predictions_aggregation(data = results, models = list(predictions.keys()), n_folds = len(pipelines_dict))
        results.to_csv(self.config.test_raw_predictions_path, index=False)

        # Evaluate the predictions
        self.evaluate_predictions(data = results, models = list(predictions.keys()))

        print(Fore.GREEN + "Testing pipeline completed successfully." + Style.RESET_ALL)