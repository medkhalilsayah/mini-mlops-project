import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/model.joblib"

app = FastAPI(title="Heart Disease Inference API")

# Charger le modèle au démarrage
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


# D'après ton dataset (colonnes affichées dans l'erreur précédente)
# Features = toutes les colonnes sauf la cible "num"
class HeartFeatures(BaseModel):
    id: int
    age: float
    sex: float
    dataset: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalch: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(features: HeartFeatures):
    if model is None:
        return {"error": f"Model not found at {MODEL_PATH}. Train the model first."}

    # 1 ligne => DataFrame
    X = pd.DataFrame([features.model_dump()])  # pydantic v2

    # prédiction binaire 0/1
    pred = int(model.predict(X)[0])

    return {"prediction": pred, "meaning": "1 = disease, 0 = no disease"}
