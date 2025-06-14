import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.schema import DatasetSchema

class TrainTestSplitter():
    def __init__(self, test_size: float):
        self.test_size = test_size

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if DatasetSchema.TARGET not in df.columns:
            raise ValueError(f"Target column '{DatasetSchema.TARGET}' not found in DataFrame.")
        
        # Split the DataFrame into train and test sets using stratified sampling to maintain the distribution of the target variable
        train, test = train_test_split(
            df, test_size=self.test_size, random_state=42, stratify=df[DatasetSchema.TARGET]
        )
        
        return train, test