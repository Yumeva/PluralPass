from __future__ import annotations

import pytest

from pluralpass.data.preprocess import _future_labels
from pluralpass.geometry import progressive_value


def test_spatial_progression_is_limited_to_the_configured_horizon() -> None:
    events = [
        {
            "period": 1,
            "possession": 7,
            "timestamp": "00:00:00.000",
            "team": {"id": 1},
            "type": {"name": "Pass"},
            "location": [20.0, 40.0],
        },
        {
            "period": 1,
            "possession": 7,
            "timestamp": "00:00:05.000",
            "team": {"id": 1},
            "type": {"name": "Carry"},
            "location": [40.0, 40.0],
        },
        {
            "period": 1,
            "possession": 7,
            "timestamp": "00:00:20.000",
            "team": {"id": 1},
            "type": {"name": "Shot"},
            "location": [110.0, 40.0],
            "shot": {"outcome": {"name": "Goal"}},
        },
    ]

    shot_within_horizon, goal_in_possession, value_delta = _future_labels(
        events, index=0, horizon=10.0
    )

    expected = progressive_value([40.0, 40.0]) - progressive_value([20.0, 40.0])
    assert shot_within_horizon == 0
    assert goal_in_possession == 1
    assert value_delta == pytest.approx(expected)
