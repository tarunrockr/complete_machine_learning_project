import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

import pickle

def save_object(file_obj, path):

    try:
        directory_name = os.path.dirname(path)
        os.makedirs(directory_name, exist_ok=True)

        pickle.dump(file_obj, open(path, 'wb'))
        logging.info("Column transfer object saved.")

    except Exception as e:
        raise  CustomException(e, sys)
