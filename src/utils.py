from copyreg import pickle
import os
import pandas as pd
import sys
import numpy as np
from src.exception import CustomException
import dill

def load_data():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(repo_root, 'component', 'archive (1)', 'data.csv'),
        os.path.join(repo_root, 'data', 'breast_cancer.csv'),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return pd.read_csv(path)

    raise FileNotFoundError(
        f"Dataset not found. Tried: {candidate_paths[0]} and {candidate_paths[1]}"
    )

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
