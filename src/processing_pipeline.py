from src.config_loaders.processing_config_loader import ProcessingConfig
from src.preprocessing.data_processor import DataPreprocessor
from src.preprocessing.train_test_splitter import TrainTestSplitter
from colorama import Fore, Style
from src.utils.schema import DatasetSchema
from src.base_pipeline import BasePipeline

class ProcessingPipeline(BasePipeline):
    
    """
    A pipeline that handles data preprocessing and preparation for training and evaluation.

    This pipeline performs the following key steps:
    1. Loads and preprocesses raw input data.
    2. Splits the preprocessed data into training and testing sets, ensuring no data leakage.
    3. Saves the resulting datasets.

    Attributes:
        config (ProcessingConfig): Configuration object containing input/output paths and split settings.
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__(config)

    def run(self):
        print(Fore.YELLOW + "Running processing pipeline..." + Style.RESET_ALL)

        # Load and preprocess the data
        data = DataPreprocessor(input_data_filename=self.config.input_data_filename).transform()

        # Perform train-test split to keep test data separate for evaluation in real-world scenarios (unseen data)
        train, test = TrainTestSplitter(test_size=self.config.test_size).transform(data)
        intersection = set(train[DatasetSchema.ID]) & set(test[DatasetSchema.ID])
        assert len(intersection) == 0, f"Some IDs appear in both train and test sets: {intersection}"
        
        # Save the processed data to CSV files
        train.to_csv(self.config.train_processed_data_filename, index=False)
        test.to_csv(self.config.test_processed_data_filename, index=False)
        print(Fore.GREEN + "Processing pipeline completed successfully" + Style.RESET_ALL)