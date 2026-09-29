"""Shared test data: a small fake customer table that follows the real format.

Unit tests must not need the internet or the real data file, so they
run in seconds on any machine, including GitHub's.
"""

import numpy as np
import pandas as pd
import pytest

SERVICE = ["Yes", "No", "No internet service"]


def fake_raw(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pick = lambda options: rng.choice(options, n)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(18, 118, n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "   # the real file has blanks exactly like this
    df = pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "gender": pick(["Male", "Female"]),
        "SeniorCitizen": pick([0, 1]),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": pick(["Yes", "No"]),
        "MultipleLines": pick(["Yes", "No", "No phone service"]),
        "InternetService": pick(["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": pick(SERVICE), "OnlineBackup": pick(SERVICE),
        "DeviceProtection": pick(SERVICE), "TechSupport": pick(SERVICE),
        "StreamingTV": pick(SERVICE), "StreamingMovies": pick(SERVICE),
        "Contract": pick(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": pick(["Yes", "No"]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    })
    # A learnable signal: short month-to-month contracts churn more
    risk = (df["Contract"] == "Month-to-month") & (df["tenure"] < 12)
    df["Churn"] = np.where(risk | (rng.random(n) < 0.1), "Yes", "No")
    return df


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return fake_raw()


@pytest.fixture
def params() -> dict:
    return {
        "split": {"test_size": 0.2, "val_size": 0.2, "seed": 42},
        "model": {"type": "logreg", "logreg": {"C": 1.0},
                  "hgb": {"learning_rate": 0.1, "max_iter": 50, "max_leaf_nodes": 15}},
        "cost": {"false_negative": 300, "false_positive": 60},
    }
