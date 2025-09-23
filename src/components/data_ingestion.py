import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer   # <-- assure-toi que ce fichier existe et contient la classe ModelTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Lecture du dataset
            df = pd.read_csv(r"C:\Users\HP\Desktop\mlproject\notebook\StudentsPerformance.csv")
            logging.info('Read the dataset as dataframe')

            # ✅ Remplacer uniquement les espaces par des underscores
            df.columns = df.columns.str.replace(" ", "_")
            logging.info("Renamed dataset columns (spaces → underscores)")

            # Création du dossier si inexistant
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Sauvegarde des données brutes
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train-test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Sauvegarde des fichiers train et test
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Étape 1 : Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Étape 2 : Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    # Étape 3 : Entraînement
    model_trainer = ModelTrainer()
    best_model_name, best_score, model_path = model_trainer.initiate_model_trainer(
        train_arr, test_arr, preprocessor_path
    )

    print(f"✅ Best model: {best_model_name} with R² = {best_score}")
    print(f"📂 Saved at: {model_path}")
