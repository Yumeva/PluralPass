from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pluralpass.config import load_config
from pluralpass.geometry import line_blockage, mirror_touchline, normalise_location, polygon_area

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "b0bc9f22dd77c206ddedc1d742893b3bbe64baec"


def test_configuration_is_pinned_to_one_upstream_commit() -> None:
    yaml_config = load_config(ROOT / "configs" / "base.yaml")
    json_config = load_config(ROOT / "configs" / "base.json")
    assert yaml_config["data"]["upstream_ref"] == UPSTREAM_COMMIT
    assert json_config["data"]["upstream_ref"] == UPSTREAM_COMMIT
    assert yaml_config["model"]["node_features"] == 14
    yaml_payload = {key: value for key, value in yaml_config.items() if not key.startswith("_")}
    json_payload = {key: value for key, value in json_config.items() if not key.startswith("_")}
    assert yaml_payload == json_payload


def test_geometry_helpers() -> None:
    assert normalise_location((60, 40)) == pytest.approx((0.0, 0.0))
    assert polygon_area([0, 0, 2, 0, 2, 2, 0, 2]) == pytest.approx(4.0)
    nodes = np.asarray([[[0.2, 0.3], [0.5, -0.7]]])
    mirrored = mirror_touchline(nodes)
    assert mirrored[..., 0] == pytest.approx(nodes[..., 0])
    assert mirrored[..., 1] == pytest.approx(-nodes[..., 1])
    opponents = [{"location": [5.0, 0.5]}]
    assert line_blockage([0, 0], [10, 0], opponents, corridor_m=1.0) == 1.0
