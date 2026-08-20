from __future__ import annotations

import numpy as np


def finite_sample_quantile(scores: np.ndarray, coverage: float) -> float:
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration scores are empty")
    level = min(np.ceil((n + 1) * coverage) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


class AdaptivePredictionSet:
    """Split-conformal APS for variable-cardinality receiver recommendations."""

    def __init__(self, coverage: float = 0.90, max_set_size: int | None = None):
        self.coverage = coverage
        self.max_set_size = max_set_size
        self.qhat: float | None = None

    @staticmethod
    def _score(probability: np.ndarray, label: int) -> float:
        order = np.argsort(-probability)
        rank = int(np.where(order == label)[0][0])
        return float(probability[order[: rank + 1]].sum())

    def fit(self, probability: np.ndarray, labels: np.ndarray) -> AdaptivePredictionSet:
        scores = np.asarray([self._score(p, int(y)) for p, y in zip(probability, labels)])
        self.qhat = finite_sample_quantile(scores, self.coverage)
        return self

    def predict(
        self, probability: np.ndarray, candidate_mask: np.ndarray | None = None
    ) -> list[np.ndarray]:
        if self.qhat is None:
            raise RuntimeError("Call fit before predict")
        sets = []
        for i, row in enumerate(probability):
            valid = (
                np.ones_like(row, dtype=bool)
                if candidate_mask is None
                else candidate_mask[i].astype(bool)
            )
            order = np.asarray([j for j in np.argsort(-row) if valid[j]], dtype=int)
            cumulative = np.cumsum(row[order])
            size = int(np.searchsorted(cumulative, self.qhat, side="left") + 1)
            if self.max_set_size is not None:
                size = min(size, self.max_set_size)
            sets.append(order[:size])
        return sets


class RegularizedAdaptivePredictionSet(AdaptivePredictionSet):
    """Regularized APS (RAPS) for set-size sensitivity analyses.

    The conformity score adds a small cardinality penalty after the first
    ``k_regularization`` sorted candidates. This keeps the same split-conformal
    machinery as APS while testing whether similarly reliable but smaller
    recommendation sets are available.
    """

    def __init__(
        self,
        coverage: float = 0.90,
        lambda_regularization: float = 0.01,
        k_regularization: int = 3,
        max_set_size: int | None = None,
    ):
        super().__init__(coverage=coverage, max_set_size=max_set_size)
        if lambda_regularization < 0:
            raise ValueError("lambda_regularization must be non-negative")
        if k_regularization < 1:
            raise ValueError("k_regularization must be at least 1")
        self.lambda_regularization = float(lambda_regularization)
        self.k_regularization = int(k_regularization)

    def _penalty(self, set_size: np.ndarray | int) -> np.ndarray | float:
        return self.lambda_regularization * np.maximum(
            np.asarray(set_size) - self.k_regularization, 0
        )

    def _score(self, probability: np.ndarray, label: int) -> float:
        order = np.argsort(-probability)
        rank = int(np.where(order == label)[0][0])
        set_size = rank + 1
        return float(probability[order[:set_size]].sum() + self._penalty(set_size))

    def predict(
        self, probability: np.ndarray, candidate_mask: np.ndarray | None = None
    ) -> list[np.ndarray]:
        if self.qhat is None:
            raise RuntimeError("Call fit before predict")
        sets = []
        for i, row in enumerate(probability):
            valid = (
                np.ones_like(row, dtype=bool)
                if candidate_mask is None
                else candidate_mask[i].astype(bool)
            )
            order = np.asarray([j for j in np.argsort(-row) if valid[j]], dtype=int)
            cumulative = np.cumsum(row[order])
            set_sizes = np.arange(1, len(order) + 1)
            regularized = cumulative + self._penalty(set_sizes)
            size = int(np.searchsorted(regularized, self.qhat, side="left") + 1)
            size = min(size, len(order))
            if self.max_set_size is not None:
                size = min(size, self.max_set_size)
            sets.append(order[:size])
        return sets


def pareto_front(success: np.ndarray, value: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    candidates = np.where(candidate_mask)[0]
    keep = []
    for i in candidates:
        dominated = any(
            (success[j] >= success[i] and value[j] >= value[i])
            and (success[j] > success[i] or value[j] > value[i])
            for j in candidates
        )
        if not dominated:
            keep.append(i)
    return np.asarray(keep, dtype=int)


def should_abstain(
    epistemic_score: float,
    threshold: float,
    visible_area_fraction: float,
    minimum_area: float = 0.25,
) -> bool:
    return bool(epistemic_score > threshold or visible_area_fraction < minimum_area)
