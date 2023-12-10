import numpy as np
import pandas as pd
import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split

class DataIngestion:

    TRAIN_DATA_PATH = os.path.join('artifacts', 'train.csv')
    TEST_DATA_PATH  = os.path.join('artifacts', 'test.csv')
    RAW_DATA_PATH   = os.path.join('artifacts', 'data.csv')

    # Getting data from csv, database api or from any other source
    def start_ingestion(self):

        try:
            logging.info("Reading the raw data csv file for data ingestion.")
            df = pd.read_csv('notebook/data/student.csv')

            # Creating a directory for the artifact(train.csv, test.csv, data.csv) files
            os.makedirs(os.path.dirname(DataIngestion.TRAIN_DATA_PATH), exist_ok=True)

            # Creating train.csv test.csv and raw.csv
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(DataIngestion.TRAIN_DATA_PATH, index=False, header=True)
            test.to_csv(DataIngestion.TEST_DATA_PATH, index=False, header=True)
            df.to_csv(DataIngestion.RAW_DATA_PATH, index=False, header=True)

            logging.info("Artifacts files saved successfully")
            # Returning train and test file path
            return DataIngestion.TRAIN_DATA_PATH, DataIngestion.TEST_DATA_PATH

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.start_ingestion()
    print("Train  file Path: ", DataIngestion.TRAIN_DATA_PATH)
    print("Test file Path:", DataIngestion.TEST_DATA_PATH)


