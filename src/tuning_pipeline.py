from src.config_loaders.tuning_config_loader import TuningConfig
import json
import os
from colorama import Fore, Style
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from src.modeling.features_selector import FeatureSelector
from sklearn.preprocessing import MinMaxScaler
import optuna
from src.utils.utils_toolbox import load_csv_data
from sklearn.pipeline import Pipeline
from src.base_pipeline import BasePipeline

class TuningPipeline(BasePipeline):

    """
    Tuning pipeline using Optuna for hyperparameter optimization.

    This class performs hyperparameter tuning using Optuna for different machine learning models 
    like RandomForest, LogisticRegression, and KNeighborsClassifier. The tuning is done using 
    cross-validation to evaluate different sets of hyperparameters based on a given scoring metric.

    Attributes:
        config (TuningConfig): Configuration for tuning process, including models to be optimized, 
                                      their hyperparameter ranges, and the Optuna settings.

    Methods:
        objective(trial, X, y, model_name, params_space, scoring, cv):
            Defines the optimization objective for Optuna. This function is called during the optimization 
            process to evaluate each trial based on the provided parameters.
        
        run():
            Runs the entire tuning pipeline, including data loading, feature selection, model tuning, 
            and saving the best hyperparameters.
    """
    
    def __init__(self, config: TuningConfig):
        super().__init__(config)
    
    @staticmethod
    def objective(trial, X, y, model_name: str, params_space: dict, scoring: str, cv: int) -> float:
        trial_params = {}
        for param_name, param_def in params_space.items():
            if param_def.type == "int":
                trial_params[param_name] = trial.suggest_int(
                    param_name,
                    param_def.low,
                    param_def.high
                )
            elif param_def.type == "float":
                trial_params[param_name] = trial.suggest_float(
                    param_name,
                    param_def.low,
                    param_def.high
                )
            elif param_def.type == "categorical":
                trial_params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_def.choices
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_def.type}")

        if model_name == "RandomForest":
            model = RandomForestClassifier(**trial_params, random_state=42)
        elif model_name == "LogisticRegression":
            model = LogisticRegression(**trial_params, random_state=42, max_iter=1000)
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier(**trial_params)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Wrap model in a pipeline with a scaler to prevent data leakage (i.e., scaling using all the data before CV)
        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", model)
        ])

        # CV
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(estimator=pipeline, X=X, y=y, cv=skf, scoring=scoring)
        return scores.mean()

    def run(self):
        print(Fore.YELLOW + "Running tuning pipeline..." + Style.RESET_ALL)

        # Load processed data
        df = load_csv_data(data_path=self.config.train_processed_data_filename)

        # Feature selection
        selector = FeatureSelector(features_selector_config=self.config.features_selector)
        X, y = selector.transform(df)

        # Optuna Tuning
        results = {} # Dictionary to store the results of tuning
        for model_name, params_space in self.config.optuna.models.items():
            print(f"{Fore.BLUE}Tuning {model_name}{Style.RESET_ALL}")

            study = optuna.create_study(direction=self.config.optuna.direction)
            study.optimize(
                lambda trial: self.objective(
                    trial,
                    X,
                    y,
                    model_name,
                    params_space,
                    self.config.optuna.scoring,
                    self.config.optuna.cv
                ),
                n_trials=self.config.optuna.n_trials
            )

            # Save the best trial parameters and the best score
            best_trial = study.best_trial
            results[model_name] = {
                "best_params": best_trial.params,
                "best_score": best_trial.value
            }

            print(f"Best validation score for {model_name}: {best_trial.value}")
            print(f"Best parameters for {model_name}: {best_trial.params}")
        
        # Save the best hyperparameters to a json file (useful for training)
        os.makedirs(os.path.dirname(self.config.best_hyperparams_filename), exist_ok=True)
        with open(self.config.best_hyperparams_filename, "w") as json_file:
            json.dump(results, json_file, indent=4)

        print(Fore.GREEN + "Tuning pipeline completed successfully." + Style.RESET_ALL)
        