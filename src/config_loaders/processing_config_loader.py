import json
from pydantic import BaseModel, Field

class ProcessingConfig(BaseModel):
    
    """"
    Configuration for data processing
    """
    input_data_filename: str = Field(..., description="Path to the input data file")
    train_processed_data_filename: str = Field(..., description="Path to the train processed data file")
    test_processed_data_filename: str = Field(..., description="Path to the test processed data file")
    test_size: float = Field(..., description="Proportion of the dataset to include in the test split")

def processing_config_loader(config_path: str) -> ProcessingConfig:
    with open(config_path, "r") as file:
        config = json.load(file)
    return ProcessingConfig(**config)