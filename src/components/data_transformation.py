import os
import sys
import numpy as np
import  pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion
from src.components.train_model import ModelTrain

class DatTransformation(object):

    # Define class variables
    PIPELINE_OBJECT_PATH = os.path.join('artifacts', 'pipeline_obj.pkl')

    def prepare_column_transformer_obj(self):

        try:
            df = pd.read_csv('notebook/data/student.csv')
            # numerical_cols   = [ col for col in  df.columns if df[col].dtype != "O" ]
            numerical_cols     = ['reading_score', 'writing_score']
            categorical_cols   = [ col for col in  df.columns if df[col].dtype == "O" ]

            # print("Num cols: ", numerical_cols)
            # print("Cat columns: ",categorical_cols)
            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            logging.info("Building numerical pipeline")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler() )
                ]
            )
            logging.info("Building categorical pipeline")
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Building column transformer object")
            col_trn_obj = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols)
                ]
            )

            logging.info("Column transformer object created")
            return  col_trn_obj

        except Exception as e:
            raise CustomException(e, sys)

    def transform_input_features(self, train_file_path, test_file_path):

        try:
            # Creating train and test dataframe
            train_file = pd.read_csv(train_file_path, index_col=False)
            test_file  = pd.read_csv(test_file_path, index_col=False)
            # print(train_file.head(5))

            col_trans_obj = self.prepare_column_transformer_obj()

            # Separating input or independent columns from the training dataframe
            train_df_input_columns = train_file.drop("math_score", axis=1)
            train_df_output_column = train_file["math_score"]

            test_df_input_columns = test_file.drop("math_score", axis=1)
            test_df_output_column = test_file["math_score"]

            train_df_transformed_input_columns = col_trans_obj.fit_transform(train_df_input_columns)
            test_df_transformed_input_columns  = col_trans_obj.transform(test_df_input_columns)

            train_array = np.concatenate((train_df_transformed_input_columns, np.array(train_df_output_column).reshape(800,1)), axis=1)
            test_array  = np.concatenate((test_df_transformed_input_columns, np.array(test_df_output_column).reshape(200, 1)), axis=1)
            logging.info("Transformation completed.")

            # Saving column transformer object into artifacts folder
            save_object(col_trans_obj, DatTransformation.PIPELINE_OBJECT_PATH)

            return train_array, test_array, col_trans_obj

        except Exception as e:
            raise

if __name__ == "__main__":

    obj1 = DataIngestion()
    train_path, test_path = obj1.start_ingestion()

    obj = DatTransformation()
    train_arr, test_arr,_ = obj.transform_input_features(train_path, test_path)

    obj2 = ModelTrain()
    best_model, r2_score = obj2.train_models(train_arr, test_arr)
    print("Best Model: ", best_model)
    print("Best Score: ", r2_score)