import json
from pathlib import Path

import numpy as np
import pytest

from churnops import metrics
from churnops.config import FEATURES, TARGET
from churnops.data import clean
from churnops.features import build_pipeline


@pytest.mark.parametrize("kind", ["logreg", "hgb"])
def test_pipeline_trains_and_gives_probabilities(raw_df, params, kind):
    df = clean(raw_df)
    params["model"]["type"] = kind
    pipe = build_pipeline(params["model"]).fit(df[FEATURES], df[TARGET])
    proba = pipe.predict_proba(df[FEATURES])[:, 1]
    assert proba.shape == (len(df),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_pipeline_survives_unseen_category(raw_df, params):
    df = clean(raw_df)
    pipe = build_pipeline(params["model"]).fit(df[FEATURES], df[TARGET])
    row = df[FEATURES].head(1).copy()
    row["PaymentMethod"] = "Crypto"   # a value never seen in training
    assert pipe.predict_proba(row).shape == (1, 2)


def test_threshold_follows_business_cost():
    y = np.array([0, 0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.9])
    cheap_misses = metrics.pick_threshold(y, proba, {"false_negative": 1, "false_positive": 100})
    costly_misses = metrics.pick_threshold(y, proba, {"false_negative": 100, "false_positive": 1})
    assert costly_misses < cheap_misses   # missing churners is expensive -> contact more people


REPORT = Path("reports/metrics.json")


@pytest.mark.skipif(not REPORT.exists(), reason="run `dvc repro` first")
def test_trained_model_passes_quality_gate():
    scores = json.loads(REPORT.read_text())
    assert scores["roc_auc"] >= 0.80
