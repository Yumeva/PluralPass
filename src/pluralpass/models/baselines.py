from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight


class ProgressiveHeuristic:
    """Rank visible teammates by forward progress with pressure/blockage penalties."""

    def score(self, nodes: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
        score = nodes[:, 0] - 0.35 * nodes[:, 9] - 0.25 * nodes[:, 10]
        return np.where(candidate_mask, score, -np.inf)


class CandidateChoiceBaseline:
    """Candidate-level pointwise baseline used only for ranking within an event."""

    def __init__(self, nonlinear: bool = False, random_state: int = 20260623):
        self.model = (
            HistGradientBoostingClassifier(max_iter=200, max_depth=6, random_state=random_state)
            if nonlinear
            else LogisticRegression(max_iter=1000, random_state=random_state)
        )

    def fit(
        self, candidate_features: np.ndarray, selected: np.ndarray
    ) -> CandidateChoiceBaseline:
        sample_weight = compute_sample_weight(class_weight="balanced", y=selected)
        self.model.fit(candidate_features, selected, sample_weight=sample_weight)
        return self

    def predict_score(self, candidate_features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(candidate_features)[:, 1]
