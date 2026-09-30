"""Pipeline stage 3: train, choose the threshold, log everything to MLflow."""

import hashlib

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from churnops import metrics
from churnops.config import FEATURES, MODEL_FILE, PROCESSED, RAW_CSV, TARGET
from churnops.features import build_pipeline

TREE_TYPE = "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    return df[FEATURES], df[TARGET]


def file_md5(path) -> str:
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def train(params: dict) -> dict:
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    model_params, cost = params["model"], params["cost"]

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment"])

    with mlflow.start_run(run_name=model_params["type"]) as run:
        # What went in: settings, and a fingerprint of the exact data file
        mlflow.log_params({"model_type": model_params["type"], **model_params[model_params["type"]]})
        mlflow.log_params({f"cost_{k}": v for k, v in cost.items()})
        mlflow.set_tag("data_md5", file_md5(RAW_CSV))
        mlflow.log_param("train_rows", len(X_train))

        pipe = build_pipeline(model_params)
        pipe.fit(X_train, y_train)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        threshold = metrics.pick_threshold(y_val, val_proba, cost)
        val_scores = metrics.compute(y_val, val_proba, threshold, cost)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_scores.items()})

        # What came out: the model, stored twice on purpose.
        # 1) in the MLflow registry, so the team can compare and promote versions;
        # 2) as one plain file, which is what DVC tracks and the API/Docker load.
        mlflow.sklearn.log_model(
            pipe, name="model",
            registered_model_name=params["mlflow"]["registered_model"],
            input_example=X_train.head(3),
            # MLflow saves with "skops", a format that refuses to load code it
            # does not know. We built this tree type ourselves, so we trust it.
            skops_trusted_types=[TREE_TYPE] if model_params["type"] == "hgb" else None,
        )
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": pipe, "threshold": threshold,
             "run_id": run.info.run_id, "model_type": model_params["type"]},
            MODEL_FILE,
        )
    return {"run_id": run.info.run_id, "threshold": threshold, **val_scores}