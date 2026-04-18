import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
import src.exception as exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    label_encoder_obj_file_path = os.path.join('artifacts', 'label_encoder.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                'radius_mean',
                'texture_mean',
                'perimeter_mean',
                'area_mean',
                'smoothness_mean'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns)
                ],
                remainder='drop'
            )

            return preprocessor
        except Exception as e:
            raise exception.CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
            try:
                train_df = pd.read_csv(train_path)      
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")

                logging.info("Obtaining preprocessing object")

                preprocessing_obj = self.get_data_transformer_object()

                target_column_name = 'diagnosis'

                input_feature_train_df = train_df.drop(columns=[target_column_name])
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name])
                target_feature_test_df = test_df[target_column_name]

                logging.info("Applying preprocessing object on training and testing dataframe")

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                # Encode target labels
                label_encoder = LabelEncoder()
                target_feature_train_arr = label_encoder.fit_transform(target_feature_train_df)
                target_feature_test_arr = label_encoder.transform(target_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
                test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

                logging.info("Saved preprocessing object.")

                save_object(
                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj
                )

                save_object(
                    file_path=self.data_transformation_config.label_encoder_obj_file_path,
                    obj=label_encoder
                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                    self.data_transformation_config.label_encoder_obj_file_path
                )
            except Exception as e:
                raise exception.CustomException(e, sys)