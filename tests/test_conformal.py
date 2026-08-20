from __future__ import annotations

import numpy as np
import pytest

from pluralpass.uncertainty.conformal import (
    AdaptivePredictionSet,
    RegularizedAdaptivePredictionSet,
    finite_sample_quantile,
    pareto_front,
    should_abstain,
)


def test_finite_sample_quantile_uses_higher_correction() -> None:
    scores = np.asarray([0.2, 0.4, 0.6, 0.8])
    assert finite_sample_quantile(scores, 0.75) == pytest.approx(0.8)


def test_aps_respects_candidate_mask() -> None:
    calibration_probability = np.asarray(
        [
            [0.60, 0.30, 0.10],
            [0.50, 0.35, 0.15],
            [0.55, 0.25, 0.20],
        ]
    )
    labels = np.asarray([0, 1, 0])
    aps = AdaptivePredictionSet(coverage=2 / 3).fit(calibration_probability, labels)
    predicted = aps.predict(
        np.asarray([[0.70, 0.20, 0.10]]),
        candidate_mask=np.asarray([[True, False, True]]),
    )[0]
    assert 1 not in predicted
    assert set(predicted).issubset({0, 2})


def test_raps_does_not_return_more_than_available_candidates() -> None:
    probability = np.asarray([[0.45, 0.35, 0.20], [0.50, 0.30, 0.20]])
    labels = np.asarray([1, 0])
    raps = RegularizedAdaptivePredictionSet(
        coverage=0.5, lambda_regularization=0.01, k_regularization=2
    ).fit(probability, labels)
    predicted = raps.predict(probability, np.ones_like(probability, dtype=bool))
    assert all(1 <= len(item) <= probability.shape[1] for item in predicted)


def test_pareto_front_and_abstention() -> None:
    success = np.asarray([0.8, 0.7, 0.9, 0.2])
    value = np.asarray([0.4, 0.6, 0.5, 0.1])
    front = pareto_front(success, value, np.asarray([True, True, True, False]))
    assert set(front) == {1, 2}
    assert should_abstain(0.6, 0.5, 0.8)
    assert should_abstain(0.1, 0.5, 0.2, minimum_area=0.25)
    assert not should_abstain(0.1, 0.5, 0.8)

