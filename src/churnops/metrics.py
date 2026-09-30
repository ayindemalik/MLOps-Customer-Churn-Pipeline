"""Scores, and the decision threshold chosen by business cost."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_cost(y, proba, threshold: float, cost: dict) -> float:
    """Average dollars lost per customer if we act on this threshold."""
    y = np.asarray(y)
    pred = np.asarray(proba) >= threshold
    fn = np.sum((y == 1) & ~pred)   # churners we did not contact
    fp = np.sum((y == 0) & pred)    # offers sent to loyal customers
    return float((fn * cost["false_negative"] + fp * cost["false_positive"]) / len(y))


def pick_threshold(y, proba, cost: dict) -> float:
    """The cut-off that loses the least money on the validation set."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = [expected_cost(y, proba, t, cost) for t in grid]
    return float(grid[int(np.argmin(costs))])


def compute(y, proba, threshold: float, cost: dict) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "cost_per_customer": expected_cost(y, proba, threshold, cost),
    }
