from src.config_loaders.processing_config_loader import ProcessingConfig
from src.preprocessing.data_processor import DataPreprocessor
from src.preprocessing.train_test_splitter import TrainTestSplitter
from colorama import Fore, Style

class ProcessingPipeline():
    
    """
    A pipeline that handles data preprocessing and preparation for training and evaluation.

    This pipeline performs the following key steps:
    1. Loads and preprocesses raw input data.
    2. Splits the preprocessed data into training and testing sets.
    3. Saves the resulting datasets.

    Attributes:
        processing_config (ProcessingConfig): Configuration object containing input/output paths and split settings.
    """

    def __init__(self, processing_config: ProcessingConfig):
        self.processing_config = processing_config

    def run(self):
        print(Fore.YELLOW + "Running processing pipeline..." + Style.RESET_ALL)

        # Load and preprocess the data
        data = DataPreprocessor(input_data_filename=self.processing_config.input_data_filename).transform()

        # Perform train-test split
        train, test = TrainTestSplitter(test_size=self.processing_config.test_size).transform(data)
        
        # Save the processed data to CSV files
        train.to_csv(self.processing_config.train_processed_data_filename, index=False)
        test.to_csv(self.processing_config.test_processed_data_filename, index=False)
        print(Fore.GREEN + "Processing pipeline completed successfully" + Style.RESET_ALL)