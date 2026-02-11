import os

# Dataset
DATA_PATH = os.getenv("DATA_PATH", "data/tweets.csv")
TEXT_COL = os.getenv("TEXT_COL", "text")          # adapte au CSV
LABEL_COL = os.getenv("LABEL_COL", "label")       # binaire: 1 = négatif, 0 = non négatif

# Split
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VAL_SIZE = float(os.getenv("VAL_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")  # local par défaut
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "airparadis_sentiment")
REGISTER_MODEL = os.getenv("REGISTER_MODEL", "0") == "1"          # option registry

# Noms modèles (registry)
MODEL_NAME_BASELINE = "airparadis_baseline"
MODEL_NAME_ADV_DL = "airparadis_advanced_dl"
MODEL_NAME_BERT = "airparadis_bert"
