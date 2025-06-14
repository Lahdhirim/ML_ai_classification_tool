import pandas as pd
from colorama import Fore

def load_csv_data(data_path: str) -> pd.DataFrame:
        try :
            data = pd.read_csv(data_path)
        except FileNotFoundError:
            raise print(Fore.RED + f"Error: {data_path} not found.")
        return data