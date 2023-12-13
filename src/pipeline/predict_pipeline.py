import pandas as pd
import numpy as np
import os
import sys

from src.exception import CustomException
from src.logger import logging
from src.utils import get_object

class ProcessInputData:

    def __init__(self, gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score):
        self.gender                      = gender
        self.race_ethnicity              = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch                       = lunch
        self.test_preparation_course     = test_preparation_course
        self.reading_score               = reading_score
        self.writing_score               = writing_score

    def get_input_dataframe(self):

        try:
            df = pd.DataFrame({
                        'gender': [self.gender],
                        'race_ethnicity': [self.race_ethnicity],
                        'parental_level_of_education': [self.parental_level_of_education],
                        'lunch': [self.lunch],
                        'test_preparation_course': [self.test_preparation_course],
                        'reading_score': [self.reading_score],
                        'writing_score': [self.writing_score]
                        })
            return df
        except Exception as e:
            raise CustomException(e, sys)

class PredictPipeline:

    def __init__(self):
        pass

    def predict_output(self, input_df):
        try:
            # Model and Column transformer path
            model_path    = os.path.join('artifacts', 'model.pkl')
            col_trns_path = os.path.join('artifacts', 'pipeline_obj.pkl')

            # Fetch file object
            column_trnsformer_obj = get_object(col_trns_path)
            model                 = get_object(model_path)

            # Applying column transformers
            transformed_input_df = column_trnsformer_obj.transform(input_df)
            # Predict input data
            predicted_data = model.predict(transformed_input_df)
            return predicted_data

        except Exception as e:
            raise CustomException(e, sys)

