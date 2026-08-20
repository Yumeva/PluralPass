from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def normalise_location(
    location: Iterable[float], length: float = 120.0, width: float = 80.0
) -> tuple[float, float]:
    x, y = (float(v) for v in location)
    return (2.0 * x / length - 1.0, 2.0 * y / width - 1.0)


def mirror_touchline(nodes: np.ndarray, y_index: int = 1) -> np.ndarray:
    mirrored = nodes.copy()
    mirrored[..., y_index] *= -1.0
    return mirrored


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return float(math.hypot(ax - bx, ay - by))


def nearest_opponent_distance(index: int, frame: list[dict]) -> float:
    actor = frame[index]
    opponents = [p for p in frame if p["teammate"] != actor["teammate"]]
    if not opponents:
        return 120.0
    return min(euclidean(actor["location"], p["location"]) for p in opponents)


def line_blockage(
    passer: Iterable[float], target: Iterable[float], opponents: list[dict], corridor_m: float = 2.0
) -> float:
    """Count opponents whose projection lies inside a pass corridor."""
    a = np.asarray(tuple(passer), dtype=float)
    b = np.asarray(tuple(target), dtype=float)
    ab = b - a
    norm = float(np.dot(ab, ab))
    if norm <= 1e-9:
        return 0.0
    blocked = 0
    for opponent in opponents:
        p = np.asarray(opponent["location"], dtype=float)
        t = float(np.dot(p - a, ab) / norm)
        if 0.0 < t < 1.0:
            distance = float(np.linalg.norm(p - (a + t * ab)))
            blocked += distance <= corridor_m
    return float(blocked)


def polygon_area(flat_polygon: list[float]) -> float:
    if len(flat_polygon) < 6:
        return 0.0
    points = np.asarray(flat_polygon, dtype=float).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def progressive_value(
    location: Iterable[float], length: float = 120.0, width: float = 80.0
) -> float:
    """A documented proxy, not a learned xT model."""
    x, y = (float(v) for v in location)
    goal_distance = math.hypot(length - x, width / 2.0 - y)
    max_distance = math.hypot(length, width / 2.0)
    centrality = 1.0 - min(abs(y - width / 2.0) / (width / 2.0), 1.0)
    return float(0.75 * (1.0 - goal_distance / max_distance) + 0.25 * centrality)
