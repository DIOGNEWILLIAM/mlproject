import os
import sys
import numpy as np
import pandas as pd  
import dill
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(x_train, y_train, x_test, y_test, models, params: dict):
    """
    Évalue plusieurs modèles de régression et retourne un dictionnaire
    contenant leurs performances (R² score).
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params.get(model_name, {})   # récupération des hyperparamètres

            # GridSearchCV pour chercher les meilleurs paramètres
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1, verbose=1, refit=True)
            gs.fit(x_train, y_train)

            best_model = gs.best_estimator_

            # prédictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
