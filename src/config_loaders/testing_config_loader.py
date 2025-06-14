import json
from pydantic import BaseModel, Field

class TestingConfig(BaseModel):
    test_processed_data_filename: str = Field(..., description="Path to the test processed data file")
    saved_models_path: str = Field(..., description="Path to the saved models directory")
    probability_threshold: float = Field(..., description="Probability threshold for aggregated prediction.")
    test_raw_predictions_path: str = Field(..., description="Path to save test raw predictions.")
    test_kpis_path: str = Field(..., description="Filename to save test KPIs.")

def testing_config_loader(config_path: str) -> TestingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return TestingConfig(**config)
