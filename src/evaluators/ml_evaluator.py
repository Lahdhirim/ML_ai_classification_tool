import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from typing import List, Dict
from src.config_loaders.training_config_loader import TrainingConfig
from src.utils.schema import DatasetSchema, EvaluatorSchema

class MLEvaluator:

    """
    Evaluator for computing and exporting performance metrics for multiple ML models across cross-validation folds.

    This class supports standard classification metrics (accuracy, recall, precision, F1-score) and saves
    them to an Excel file where each sheet corresponds to a metric.

    Attributes:
        config (TrainingConfig): Configuration object providing output path for validation KPIs.
        models (List[str]): List of model names for which metrics will be computed.
    
    Methods:
        evaluate(df): Computes metrics per fold and exports them to an Excel file.
    """
    
    def __init__(self, config: TrainingConfig, models: List[str]):
        self.models = models
        self.validation_kpis_path = config.validation_kpis_path

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        results = {}
        y_true = df[DatasetSchema.TARGET]
        for model in self.models:
            y_pred = df[model]
            results[model] = {
                EvaluatorSchema.ACCURACY: accuracy_score(y_true, y_pred),
                EvaluatorSchema.RECALL: recall_score(y_true, y_pred),
                EvaluatorSchema.PRECISION: precision_score(y_true, y_pred),
                EvaluatorSchema.F1: f1_score(y_true, y_pred)
            }
        return results

    @staticmethod
    def result_formatter(results: Dict[str, pd.Series], kpis_path: str) -> None:
        metrics = [EvaluatorSchema.ACCURACY, EvaluatorSchema.RECALL, EvaluatorSchema.PRECISION, EvaluatorSchema.F1]
        with pd.ExcelWriter(kpis_path) as writer:
            for metric in metrics:
                df_metric = pd.DataFrame(
                    {fold: {model: values.get(metric, None) for model, values in models.items()} for fold, models in results.items()}
                ).T
                df_metric.loc[EvaluatorSchema.MEAN] = df_metric.mean() # Add mean over all folds
                df_metric.index.name = DatasetSchema.FOLD
                df_metric.to_excel(writer, sheet_name=metric)
            print(f"KPIs Results saved to {kpis_path}")

    def evaluate(self, df: pd.DataFrame) -> None:
        results_df = df.copy()
        per_fold_metrics = results_df.groupby(DatasetSchema.FOLD).apply(self._compute_metrics) # Compute metrics for each fold to ensure that folds do not impact the models performances
        self.result_formatter(results = per_fold_metrics, kpis_path = self.validation_kpis_path) # Save results to Excel file with each metric in a separate sheet