"""The model: preprocessing and classifier inside ONE scikit-learn Pipeline.

Saving the whole pipeline means the API applies exactly the same
preprocessing as training. That removes a whole family of production bugs
called "training/serving skew".
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churnops.config import CATEGORICAL, NUMERIC


def build_pipeline(model_params: dict) -> Pipeline:
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])
    kind = model_params["type"]
    if kind == "logreg":
        clf = LogisticRegression(C=model_params["logreg"]["C"], max_iter=1000, class_weight="balanced")
    elif kind == "hgb":
        hp = model_params["hgb"]
        clf = HistGradientBoostingClassifier(
            learning_rate=hp["learning_rate"], max_iter=hp["max_iter"],
            max_leaf_nodes=hp["max_leaf_nodes"], class_weight="balanced", random_state=0,
        )
    else:
        raise ValueError(f"Unknown model type: {kind!r} (use 'logreg' or 'hgb')")
    return Pipeline([("prep", prep), ("model", clf)])
