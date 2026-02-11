from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path

# ==========================================================
# Configuration logging (utile pour Azure Application Insights)
# ==========================================================
logger = logging.getLogger("airparadis")
logging.basicConfig(level=logging.INFO)

# ==========================================================
# Initialisation FastAPI
# ==========================================================
app = FastAPI(title="AirParadis Sentiment API")

# ==========================================================
# Chargement robuste du modèle
# ==========================================================

# Chemin du dossier contenant main.py
HERE = Path(__file__).resolve().parent

# Chemin vers api/model
MODEL_DIR = HERE / "model"

if not MODEL_DIR.exists():
    raise OSError(f"Model directory not found: {MODEL_DIR}")

model = tf.keras.models.load_model(str(MODEL_DIR))

# ==========================================================
# Schemas d'entrée/sortie
# ==========================================================

class TweetIn(BaseModel):
    text: str


class PredOut(BaseModel):
    negative_proba: float
    negative_label: int


class FeedbackIn(BaseModel):
    text: str
    negative_proba: float
    negative_label: int
    is_wrong: bool = True


# ==========================================================
# Routes API
# ==========================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredOut)
def predict(payload: TweetIn):
    text = (payload.text or "").strip()

    proba = float(
        model.predict(
            np.array([text], dtype=str),
            verbose=0
        ).ravel()[0]
    )

    label = int(proba >= 0.5)

    return {
        "negative_proba": proba,
        "negative_label": label
    }


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    """
    Route utilisée pour remonter les prédictions
    que l'utilisateur considère comme incorrectes.
    Cela sera analysé dans Azure Application Insights.
    """
    logger.info(
        "USER_FEEDBACK | is_wrong=%s | proba=%.4f | label=%s | text=%s",
        payload.is_wrong,
        payload.negative_proba,
        payload.negative_label,
        payload.text[:300]
    )

    return {"status": "logged"}
