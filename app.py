from flask import Flask, request, render_template
import os
import sys
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, ProcessInputData

application = Flask(__name__)
app  = application


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictions', methods = ['GET','POST'])
def data_predictions():

    if request.method == "GET":
        pred_data = []
        return render_template('prediction.html', results=pred_data)
    elif request.method == "POST":

        pred_pipe_obj = ProcessInputData(
            request.form.get('gender'),
            request.form.get('race_ethnicity'),
            request.form.get('parental_level_of_education'),
            request.form.get('lunch'),
            request.form.get('test_preparation_course'),
            request.form.get('reading_score'),
            request.form.get('writing_score')
        )
        input_df = pred_pipe_obj.get_input_dataframe()

        pipeline_obj = PredictPipeline()
        pred_data    = pipeline_obj.predict_output(input_df)

        return  render_template('prediction.html', results=pred_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)