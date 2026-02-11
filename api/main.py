from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path

logger = logging.getLogger("airparadis")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AirParadis Sentiment API")

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"

# Ne jamais casser l'import si le modèle n'existe pas (CI / prod sans modèle local)
model = None
if MODEL_DIR.exists():
    try:
        model = tf.keras.models.load_model(str(MODEL_DIR))
        logger.info("✅ Model loaded from: %s", MODEL_DIR)
    except Exception as e:
        logger.exception("❌ Failed to load model from %s: %s", MODEL_DIR, e)
        model = None
else:
    logger.warning("⚠️ Model directory not found: %s (fallback mode)", MODEL_DIR)


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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredOut)
def predict(payload: TweetIn):
    text = (payload.text or "").strip()

    # Fallback mode (CI / modèle absent) : prédiction neutre
    if model is None:
        return {"negative_proba": 0.5, "negative_label": 0}

    proba = float(model.predict(np.array([text], dtype=str), verbose=0).ravel()[0])
    label = int(proba >= 0.5)
    return {"negative_proba": proba, "negative_label": label}


@app.post("/feedback")
def feedback(payload: FeedbackIn):
    logger.info(
        "USER_FEEDBACK | is_wrong=%s | proba=%.4f | label=%s | text=%s",
        payload.is_wrong,
        payload.negative_proba,
        payload.negative_label,
        payload.text[:300],
    )
    return {"status": "logged"}
