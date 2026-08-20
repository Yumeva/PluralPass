from __future__ import annotations

from itertools import pairwise

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)


def topk_accuracy(probability: np.ndarray, labels: np.ndarray, k: int) -> float:
    top = np.argpartition(-probability, min(k - 1, probability.shape[1] - 1), axis=1)[:, :k]
    return float(np.mean([label in row for label, row in zip(labels, top)]))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lower, upper in pairwise(edges):
        membership = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if membership.any():
            total += membership.mean() * abs(y[membership].mean() - p[membership].mean())
    return float(total)


def receiver_metrics(probability: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "top1": topk_accuracy(probability, labels, 1),
        "top3": topk_accuracy(probability, labels, 3),
        "nll": float(log_loss(labels, probability, labels=np.arange(probability.shape[1]))),
    }


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
    }


def value_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    correlation = spearmanr(y, prediction).statistic
    return {"mae": float(mean_absolute_error(y, prediction)), "spearman": float(correlation)}


def set_metrics(
    sets: list[np.ndarray], labels: np.ndarray, abstained: np.ndarray | None = None
) -> dict[str, float]:
    included = np.asarray([int(label in prediction) for label, prediction in zip(labels, sets)])
    size = np.asarray([len(prediction) for prediction in sets])
    metrics = {"coverage": float(included.mean()), "mean_set_size": float(size.mean())}
    if abstained is not None:
        retained = ~abstained.astype(bool)
        metrics["abstention_rate"] = float(abstained.mean())
        metrics["selective_coverage"] = (
            float(included[retained].mean()) if retained.any() else float("nan")
        )
    return metrics
