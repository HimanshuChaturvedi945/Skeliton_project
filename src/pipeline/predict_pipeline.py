import sys
import pandas as pd
from src.utils import load_object, save_object
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            data_scaled = preprocessor.transform(features)
            prediction = model.predict(data_scaled)

            return prediction
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
                 radius_mean: float,
                 texture_mean: float,
                 perimeter_mean: float,
                 area_mean: float,
                 smoothness_mean: float):
        self.radius_mean = radius_mean
        self.texture_mean = texture_mean
        self.perimeter_mean = perimeter_mean
        self.area_mean = area_mean
        self.smoothness_mean = smoothness_mean

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "radius_mean": [self.radius_mean],
                "texture_mean": [self.texture_mean],
                "perimeter_mean": [self.perimeter_mean],
                "area_mean": [self.area_mean],
                "smoothness_mean": [self.smoothness_mean]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)
