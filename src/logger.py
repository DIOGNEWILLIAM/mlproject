import logging
import os
from datetime import datetime

def get_logger(name=__name__):
    """
    Crée et retourne un logger configurable pour le projet.
    Les logs sont écrits à la fois dans un fichier horodaté et dans la console.
    """
    # Créer le dossier logs s'il n'existe pas
    logs_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(logs_path, exist_ok=True)

    # Fichier log horodaté
    log_file = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    log_file_path = os.path.join(logs_path, log_file)

    # Création du logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Niveau par défaut, modifiable

    # Format commun
    formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

    # Handler fichier
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Exemple d'utilisation
if __name__ == "__main__":
    logging = get_logger(__name__)
    logging.info("Logger initialisé avec succès")
    logging.warning("Ceci est un warning de test")
    logging.error("Ceci est une erreur de test")
