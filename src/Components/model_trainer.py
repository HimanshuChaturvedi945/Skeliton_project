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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42)
            }

            params = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.01, 0.1],
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1.0, 10.0]
                }
            }

            if XGBOOST_AVAILABLE:
                models["XGBoost"] = XGBClassifier(eval_metric='logloss', random_state=42)
                params["XGBoost"] = {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                }

            if CATBOOST_AVAILABLE:
                models["CatBoosting"] = CatBoostClassifier(verbose=False, random_state=42)
                params["CatBoosting"] = {
                    "iterations": [50, 100],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6]
                }

            model_report: dict = {}
            best_models: dict = {}

            for model_name, model in models.items():
                logging.info(f"Starting hyperparameter tuning for {model_name}")
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params.get(model_name, {}),
                    cv=3,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)

                best_estimator = grid_search.best_estimator_
                y_test_pred = best_estimator.predict(X_test)
                test_model_score = accuracy_score(y_test, y_test_pred)

                model_report[model_name] = test_model_score
                best_models[model_name] = best_estimator

                logging.info(
                    f"{model_name}: best params={grid_search.best_params_}, test accuracy={test_model_score:.4f}"
                )

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = best_models[best_model_name]

            print(f"\n{'='*50}")
            print("MODEL TRAINING RESULTS")
            print(f"{'='*50}")
            print(f"Best Model: {best_model_name}")
            print(f"Best Accuracy Score: {best_model_score:.4f}")
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