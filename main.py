from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Telco Churn Model API")

model_path = Path(__file__).resolve().parent / "models" / "churn_model.joblib"
artefact = joblib.load(model_path)
model = artefact["model"]
threshold = artefact["threshold"]
feature_names = artefact["feature_names"]


class ChurnRequest(BaseModel):
    gender: str = Field(examples=["Female"])
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["No"])
    tenure: int = Field(ge=0, examples=[1])
    PhoneService: str = Field(examples=["No"])
    MultipleLines: str = Field(examples=["No phone service"])
    InternetService: str = Field(examples=["DSL"])
    OnlineSecurity: str = Field(examples=["No"])
    OnlineBackup: str = Field(examples=["Yes"])
    DeviceProtection: str = Field(examples=["No"])
    TechSupport: str = Field(examples=["No"])
    StreamingTV: str = Field(examples=["No"])
    StreamingMovies: str = Field(examples=["No"])
    Contract: str = Field(examples=["Month-to-month"])
    PaperlessBilling: str = Field(examples=["Yes"])
    PaymentMethod: str = Field(examples=["Electronic check"])
    MonthlyCharges: float = Field(gt=0, examples=[29.85])
    TotalCharges: float = Field(ge=0, examples=[29.85])


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: ChurnRequest):
    features = pd.DataFrame([request.model_dump()])[feature_names]

    probability = float(model.predict_proba(features)[0, 1])
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "class_name": ["stayed", "churned"][prediction],
        "churn_probability": round(probability, 4),
        "threshold": threshold,
    }
