import os
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mon_mlops_project.serving.schemas import HouseFeatures, PredictionResponse


MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "CaliforniaHousingModel")
MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/artifacts/model")



def get_model_uri() -> str:
    return f"models:/{MODEL_NAME}/{MODEL_STAGE}"


@lru_cache(maxsize=1)
def load_model():
    # 1) Priorité: modèle local embarqué dans l'image Docker
    if os.path.exists(MODEL_PATH):
        model_uri = f"file://{MODEL_PATH}"
        model = mlflow.sklearn.load_model(MODEL_PATH)
        return model_uri, model

    # 2) Sinon: fallback sur MLflow Registry (utile en dev)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_uri = get_model_uri()
    model = mlflow.sklearn.load_model(model_uri)
    return model_uri, model



# ======================
# FastAPI app
# ======================
app = FastAPI(
    title="ImmoPrix API",
    version="1.0.0",
    description="API de prédiction du prix médian des maisons (California Housing) via MLflow Model Registry.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# ROUTES
# ======================

# 👉 AJOUTE CETTE ROUTE ICI 👇
@app.get("/")
def root():
    return {"message": "ImmoPrix API is running. Go to /docs for Swagger."}


@app.get("/health")
def health():
    model_uri, _ = load_model()
    return {"status": "ok", "model_uri": model_uri}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: HouseFeatures):
    model_uri, model = load_model()

    row = payload.model_dump()
    X = pd.DataFrame(
        [row],
        columns=[
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude"
        ],
    )

    pred = float(model.predict(X)[0])
    return PredictionResponse(model_uri=model_uri, prediction=pred)
