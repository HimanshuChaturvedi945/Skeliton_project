import os
import sys
from dataclasses import dataclass

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "Logistic Regression": LogisticRegression()
            }
            
            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBClassifier()
                
            if CATBOOST_AVAILABLE:
                models["CatBoosting"] = CatBoostClassifier(verbose=False)

            model_report: dict = {}

            for i in range(len(models)):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)

                test_model_score = r2_score(y_test, y_test_pred)

                model_report[list(models.keys())[i]] = test_model_score

            best_model_score = max(model_report.values())

            best_model_name = [key for key in model_report if model_report[key] == best_model_score][0]

            best_model = models[best_model_name]

            print(f"\n{'='*50}")
            print("MODEL TRAINING RESULTS")
            print(f"{'='*50}")
            print(f"Best Model: {best_model_name}")
            print(f"Best R² Score: {best_model_score:.4f}")
            print(f"{'='*50}")
            print("All Model Scores:")
            for model_name, score in model_report.items():
                print(f"  {model_name}: {score:.4f}")
            print(f"{'='*50}\n")

            logging.info(f"Best found model on both training and testing dataset is {best_model_name} with score {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e, sys)