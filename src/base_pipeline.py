from src.config_loaders.processing_config_loader import ProcessingConfig
from src.config_loaders.training_config_loader import TrainingConfig
from src.config_loaders.testing_config_loader import TestingConfig
from abc import ABC, abstractmethod
from typing import Union

class BasePipeline(ABC):
    """Abstract base class for main pipelines in the AI Classification Tool."""
    
    def __init__(self, config: Union[ProcessingConfig, TrainingConfig, TestingConfig]):
        self.config = config
    
    @abstractmethod
    def run(self) -> None:
        """run the pipeline."""
        pass