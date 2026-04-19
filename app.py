from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    else:
        data = CustomData(
            radius_mean=float(request.form.get('radius_mean')),
            texture_mean=float(request.form.get('texture_mean')),
            perimeter_mean=float(request.form.get('perimeter_mean')),
            area_mean=float(request.form.get('area_mean')),
            smoothness_mean=float(request.form.get('smoothness_mean'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('home.html', prediction_text=f"The predicted class is: {result[0]}")
       
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)