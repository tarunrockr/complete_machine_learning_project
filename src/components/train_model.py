import os
import sys
import pandas as pd
import numpy as np
from src.exception import  CustomException
from src.logger import logging
from src.utils import save_object, train_multiple_models

# Importing models
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
# from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score


class ModelTrain:

    MODEL_FILE_PATH = os.path.join('artifacts', 'model.pkl')

    def train_models(self, train_array, test_array):
        try:
            logging.info("Model training start")

            # Separating independent and dependent features fro train and test dataset
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])

            model_dict = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                # "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor()
                # "Support Vector Regressor": SVR()
            }

            # Parameter dictionary for hyperparamer tuning
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],

                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                # "XGBRegressor": {
                #     'learning_rate': [.1, .01, .05, .001],
                #     'n_estimators': [8, 16, 32, 64, 128, 256]
                # },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "KNN": {},
                # "Support Vector Regressor": {
                #     'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                #     'gamma': ['scale', 'auto']
                # }

            }

            model_score_dict = train_multiple_models(X_train, X_test, y_train, y_test, model_dict, params)

            # Sorting dictionary in reverse order
            model_score_dict = {k:v for k,v in sorted(model_score_dict.items(), key=lambda item: item[1], reverse=True)}
            # print("Sorted dict result: ", model_score_dict)

            best_r2_score = max(list(model_score_dict.values()))
            # print("Best model R2 Score score: ", best_r2_score )

            best_model_name = [k for k,v in model_score_dict.items() if v == max(model_score_dict.values())][0]
            # print("Best_model name: ", best_model_name )

            # Saving model to artifacts
            save_object(model_dict[best_model_name], ModelTrain.MODEL_FILE_PATH)

            return best_model_name,best_r2_score

        except Exception as e:
            raise CustomException(e, sys)

