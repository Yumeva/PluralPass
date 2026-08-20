from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class XTAction:
    action_type: str
    start: tuple[float, float]
    end: tuple[float, float] | None = None
    success: bool = True
    goal: bool = False


class ExpectedThreatGrid:
    """Reproducible possession-based xT baseline fitted only on training actions.

    The implementation follows the standard move/shot decomposition. It is a
    baseline, not a claim that grid xT represents an unobserved causal outcome.
    """

    def __init__(self, x_bins: int = 16, y_bins: int = 12, iterations: int = 40):
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.iterations = iterations
        self.values = np.zeros((x_bins, y_bins), dtype=float)

    def _bin(self, location: tuple[float, float]) -> tuple[int, int]:
        x, y = location
        bx = min(max(int(x / 120.0 * self.x_bins), 0), self.x_bins - 1)
        by = min(max(int(y / 80.0 * self.y_bins), 0), self.y_bins - 1)
        return bx, by

    def fit(self, actions: Iterable[XTAction]) -> ExpectedThreatGrid:
        actions = list(actions)
        shots = np.zeros_like(self.values)
        goals = np.zeros_like(self.values)
        possession_goals = np.zeros_like(self.values)
        moves = np.zeros_like(self.values)
        all_actions = np.zeros_like(self.values)
        transition = np.zeros((self.x_bins * self.y_bins, self.x_bins * self.y_bins), dtype=float)
        for action in actions:
            start = self._bin(action.start)
            all_actions[start] += 1
            possession_goals[start] += int(action.goal)
            if action.action_type == "shot":
                shots[start] += 1
                goals[start] += int(action.goal)
            elif action.action_type in {"pass", "carry"}:
                moves[start] += 1
                if action.success and action.end is not None:
                    end = self._bin(action.end)
                    source = start[0] * self.y_bins + start[1]
                    target = end[0] * self.y_bins + end[1]
                    transition[source, target] += 1
        p_shot = np.divide(shots, all_actions, out=np.zeros_like(shots), where=all_actions > 0)
        p_move = np.divide(moves, all_actions, out=np.zeros_like(moves), where=all_actions > 0)
        p_goal = np.divide(goals, shots, out=np.zeros_like(goals), where=shots > 0)
        p_possession_goal = np.divide(
            possession_goals,
            all_actions,
            out=np.zeros_like(possession_goals),
            where=all_actions > 0,
        )
        move_attempts = moves.reshape(-1, 1)
        transition = np.divide(
            transition,
            move_attempts,
            out=np.zeros_like(transition),
            where=move_attempts > 0,
        )
        values = np.zeros(self.x_bins * self.y_bins, dtype=float)
        immediate = np.maximum(p_shot * p_goal, p_possession_goal).reshape(-1)
        move_probability = p_move.reshape(-1)
        for _ in range(self.iterations):
            values = immediate + move_probability * (transition @ values)
        self.values = values.reshape(self.x_bins, self.y_bins)
        return self

    def value(self, location: tuple[float, float]) -> float:
        return float(self.values[self._bin(location)])

    def delta(self, start: tuple[float, float], end: tuple[float, float]) -> float:
        return self.value(end) - self.value(start)


class SoccerActionVAEPAdapter:
    """Explicit optional boundary for the published socceraction implementation."""

    def __init__(self):
        try:
            import socceraction  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Install the optional baseline with `pip install -e '.[socceraction]'`; "
                "do not silently replace VAEP with the progression proxy."
            ) from error
