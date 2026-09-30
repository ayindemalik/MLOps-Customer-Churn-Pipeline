"""Pipeline stage 4: score on the untouched test set, apply the quality gate,
and promote the model to "champion" only if it beats the current one."""

import json

import joblib
import mlflow
from mlflow import MlflowClient

from churnops import metrics
from churnops.config import MODEL_FILE, REPORTS
from churnops.train import load_split


def evaluate(params: dict) -> dict:
    bundle = joblib.load(MODEL_FILE)
    X_test, y_test = load_split("test")
    proba = bundle["pipeline"].predict_proba(X_test)[:, 1]
    scores = metrics.compute(y_test, proba, bundle["threshold"], params["cost"])
    scores["threshold"] = bundle["threshold"]

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    with mlflow.start_run(run_id=bundle["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in scores.items()})

    gate = params["gate"]["min_roc_auc"]
    scores["passed_gate"] = scores["roc_auc"] >= gate
    scores["promoted"] = scores["passed_gate"] and promote(params, bundle["run_id"], scores["roc_auc"])
    return scores


def promote(params: dict, run_id: str, test_auc: float) -> bool:
    """Point the 'champion' alias at this version if it beats the current champion."""
    client = MlflowClient(tracking_uri=params["mlflow"]["tracking_uri"])
    name = params["mlflow"]["registered_model"]
    version = client.search_model_versions(f"name='{name}' and run_id='{run_id}'")[0].version
    try:
        champion = client.get_model_version_by_alias(name, "champion")
        best_auc = client.get_run(champion.run_id).data.metrics.get("test_roc_auc", 0.0)
    except mlflow.exceptions.MlflowException:
        best_auc = 0.0   # no champion yet
    if test_auc > best_auc:
        client.set_registered_model_alias(name, "champion", version)
        return True
    return False
