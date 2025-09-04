import logging
import os
from datetime import datetime

# Nom du fichier log basé sur la date/heure
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Dossier "logs" dans le projet
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Chemin complet du fichier log
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configuration du logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


