import pandas as pd
from colorama import Fore
from src.utils.utils_toolbox import load_csv_data

class DataPreprocessor():

    """
    A class for preprocessing raw input data.
    This class is responsible for loading, cleaning, and transforming the raw input data
    Attributes:
        input_data_filename (str): Path to the raw input CSV file.

    Methods:
        clean_data(data): Cleans and filters the loaded data.
        transform(): Full preprocessing pipeline returning cleaned data.
    """

    def __init__(self, input_data_filename: str):
        self.input_data_filename = input_data_filename
        
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data_copy = data.copy()

        ##############################################
        # Perform necessary cleaning steps here
        #############################################

        return data_copy

    def transform(self) -> pd.DataFrame:
        data = load_csv_data(data_path = self.input_data_filename)
        print("Shape of the original dataset: ", data.shape)
        data = self.clean_data(data)
        print("Shape of the dataset after processing: ", data.shape)
        return data