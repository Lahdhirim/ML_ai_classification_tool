class DatasetSchema:
    FEATURE_1 = "feature_1"
    FEATURE_2 = "feature_2"
    FEATURE_3 = "feature_3"
    TARGET = "target"

class PredictionSchema:
    PREDICTION_AGGREGATED = "Prediction_Aggregated"
    MLP = "MLP"
    MODEL = "Model"

class PipelinesDictSchema:
    FEATURE_SELECTOR = "features_selector"
    SCALER = "scaler"
    MODELS_PARAMS = "models_params"
    ML_MODELS = "MLModels"
    MLP_MODEL = "mlp_model"

class EvaluatorSchema:
    PRECISION = "Precision"
    RECALL = "Recall"
    F1 = "F1"
    ACCURACY = "Accuracy"
    MEAN = "Mean"