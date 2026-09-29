"""Paths, column lists and the params.yaml loader. One place for every name."""

from pathlib import Path

import yaml

RAW_CSV = Path("data/raw/telco.csv")
PROCESSED = Path("data/processed")
MODEL_FILE = Path("models/model.joblib")
REPORTS = Path("reports")

TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURES = NUMERIC + CATEGORICAL

# the params.yaml loader is used in many places, 
# so we put it here to avoid duplication
# load_params 
def load_params(path: str | Path = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
