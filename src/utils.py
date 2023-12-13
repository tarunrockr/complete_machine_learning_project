import os
import sys
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV

import pickle

def save_object(file_obj, path):

    try:
        directory_name = os.path.dirname(path)
        os.makedirs(directory_name, exist_ok=True)

        pickle.dump(file_obj, open(path, 'wb'))
        logging.info("Column transfer object saved.")

    except Exception as e:
        raise  CustomException(e, sys)

def get_object(path):
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        return  obj

    except Exception as e:
        raise CustomException(e, sys)

def train_multiple_models(X_train, X_test, y_train, y_test, model_dict, params):

    model_score_dict = {}

    for key, value in model_dict.items():
        # print("Key: ", key, " value: ", value)
        model = value
        model_param = params[key]

        grid = GridSearchCV(model, model_param, cv=3)
        grid.fit(X_train, y_train)

        print("Model Name: ", key)
        print("Best params: ", grid.best_params_)
        print("--------------------------------")
        # training model
        model.set_params(**grid.best_params_)
        model.fit(X_train, y_train)
        # Prediction on test data
        y_pred = model.predict(X_test)
        r2_scr = r2_score(y_test, y_pred)

        model_score_dict[key] = r2_scr

    return model_score_dict