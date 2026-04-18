import os
import pandas as pd

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
