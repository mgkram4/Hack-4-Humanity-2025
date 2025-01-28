import os
import pandas as pd

class DDPreprocessor:
    def __init__(self, data='plant_disease_dataset.csv'):
        current_dir = os.getcwd()  # make sure you cd to "Hack-4-Humanity-2025" directory first
        file_path = os.path.join(current_dir, 'backend', 'models', 'disease_detection', data)
        self.df = pd.read_csv(file_path)
        self.transform()

    # Prerequisite for modeling - refining data
    def transform(self) -> None:
        # Drop all invalid data
        self.df.drop_duplicates(inplace=True)
        self.df.dropna(inplace=True)
        # Convert categorical data to numericals
        self.df["Label"].replace("Powdery", 0.0, inplace=True)
        self.df["Label"].replace("Healthy", 1.0, inplace=True)
        self.df["Label"].replace("Rust", 2.0, inplace=True)
        # Convert numpy array string into a numpy array
        self.df["ImageData"] = self.df["ImageData"].values
        
# Test code
# pre = DDPreprocessor()
# print(pre.df)
