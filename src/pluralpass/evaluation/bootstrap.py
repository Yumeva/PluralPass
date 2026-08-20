from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

import numpy as np

from pluralpass.evaluation.metrics import binary_metrics, receiver_metrics, value_metrics


def _interval(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "lower": float(np.nanquantile(array, 0.025)),
        "upper": float(np.nanquantile(array, 0.975)),
    }


def cluster_bootstrap_intervals(
    *,
    groups: np.ndarray,
    receiver_probability: np.ndarray,
    labels: np.ndarray,
    completed: np.ndarray,
    chosen_completion: np.ndarray,
    value: np.ndarray,
    chosen_value: np.ndarray,
    included: np.ndarray,
    set_size: np.ndarray,
    abstained: np.ndarray,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile intervals from a match-cluster bootstrap."""
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    estimates: dict[tuple[str, str], list[float]] = defaultdict(list)

    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled_groups])

        for metric, estimate in receiver_metrics(receiver_probability[rows], labels[rows]).items():
            estimates[("receiver", metric)].append(estimate)

        if np.unique(completed[rows]).size > 1:
            for metric, estimate in binary_metrics(
                completed[rows], chosen_completion[rows]
            ).items():
                estimates[("completion", metric)].append(estimate)

        for metric, estimate in value_metrics(value[rows], chosen_value[rows]).items():
            estimates[("value", metric)].append(estimate)

        retained = ~abstained[rows]
        set_estimates = {
            "coverage": float(included[rows].mean()),
            "mean_set_size": float(set_size[rows].mean()),
            "abstention_rate": float(abstained[rows].mean()),
            "selective_coverage": (
                float(included[rows][retained].mean()) if retained.any() else float("nan")
            ),
        }
        for metric, estimate in set_estimates.items():
            estimates[("recommendation_set", metric)].append(estimate)

    result: dict[str, dict[str, dict[str, float]]] = {}
    for (section, metric), values in estimates.items():
        result.setdefault(section, {})[metric] = _interval(values)
    return result


def paired_cluster_bootstrap_differences(
    *,
    groups: np.ndarray,
    labels: np.ndarray,
    model_prediction: np.ndarray,
    baseline_prediction: np.ndarray,
    metric_function: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Intervals for paired model-minus-baseline metric differences."""
    unique_groups = np.unique(groups)
    group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    differences: dict[str, list[float]] = defaultdict(list)
    point_model = metric_function(labels, model_prediction)
    point_baseline = metric_function(labels, baseline_prediction)

    for _ in range(repetitions):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([group_rows[group] for group in sampled_groups])
        model_metrics = metric_function(labels[rows], model_prediction[rows])
        baseline_metrics = metric_function(labels[rows], baseline_prediction[rows])
        for metric in model_metrics:
            differences[metric].append(model_metrics[metric] - baseline_metrics[metric])

    return {
        metric: {
            "difference": float(point_model[metric] - point_baseline[metric]),
            **_interval(values),
        }
        for metric, values in differences.items()
    }
