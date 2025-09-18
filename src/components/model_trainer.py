import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

import warnings
warnings.filterwarnings("ignore")


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                      'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                                      'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Linear Regression": {},
                "Lasso": {},
                "Ridge": {},
                "K-Neighbors Regressor": {'n_neighbors': [5, 7, 9, 11]},
                "XGB Regressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                  'n_estimators': [8, 16, 32, 64, 128, 256]},
                "CatBoost Regressor": {'depth': [6, 8, 10],
                                       'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                       'iterations': [30, 50, 100]},
                "AdaBoost Regressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001],
                                       'n_estimators': [8, 16, 32, 64, 128, 256]}
            }

            # évalue tous les modèles et récupère les R²
            model_report: dict = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models,
                params=params
            )

            best_model_name = max(model_report, key=lambda name: model_report[name])
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found (score < 0.6)")

            logging.info(f"Best found model: {best_model_name} with R²={best_model_score}")

            # Refit du meilleur modèle avec GridSearchCV
            best_model = GridSearchCV(
                estimator=models[best_model_name],
                param_grid=params.get(best_model_name, {}),
                cv=3,
                n_jobs=-1,
                refit=True,
                verbose=1
            )
            best_model.fit(x_train, y_train)

            # Sauvegarde du modèle
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Prédictions et R²
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return best_model_name, best_model_score, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
