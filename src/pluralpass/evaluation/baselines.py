from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from pluralpass.config import artifact_run_name
from pluralpass.evaluation.bootstrap import paired_cluster_bootstrap_differences
from pluralpass.evaluation.metrics import binary_metrics, receiver_metrics, value_metrics
from pluralpass.models.baselines import CandidateChoiceBaseline, ProgressiveHeuristic
from pluralpass.models.xt import ExpectedThreatGrid, SoccerActionVAEPAdapter, XTAction
from pluralpass.training import _dataset


def _choice_data(
    rows: list[dict[str, Any]], max_nodes: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    features: list[list[float]] = []
    selected: list[int] = []
    candidates_by_event: list[np.ndarray] = []
    for row in rows:
        candidates = np.flatnonzero(np.asarray(row["candidate_mask"], dtype=bool))
        candidates = candidates[candidates < max_nodes]
        candidates_by_event.append(candidates)
        for candidate in candidates:
            features.append(row["nodes"][int(candidate)])
            selected.append(int(candidate == int(row["receiver_index"])))
    return np.asarray(features, dtype=float), np.asarray(selected), candidates_by_event


def _event_probabilities(
    scores: np.ndarray, candidates_by_event: list[np.ndarray], max_nodes: int
) -> np.ndarray:
    probability = np.zeros((len(candidates_by_event), max_nodes), dtype=float)
    offset = 0
    for event, candidates in enumerate(candidates_by_event):
        event_scores = np.clip(scores[offset : offset + len(candidates)], 1e-9, None)
        probability[event, candidates] = event_scores / event_scores.sum()
        offset += len(candidates)
    return probability


def _heuristic_probabilities(rows: list[dict[str, Any]], max_nodes: int) -> np.ndarray:
    heuristic = ProgressiveHeuristic()
    probability = np.zeros((len(rows), max_nodes), dtype=float)
    for event, row in enumerate(rows):
        nodes = np.asarray(row["nodes"][:max_nodes], dtype=float)
        candidate_mask = np.asarray(row["candidate_mask"][:max_nodes], dtype=bool)
        scores = heuristic.score(nodes, candidate_mask)
        valid_scores = scores[candidate_mask]
        weights = np.exp(valid_scores - valid_scores.max())
        probability[event, np.flatnonzero(candidate_mask)] = weights / weights.sum()
    return probability


def _denormalise_node_location(
    node: list[float] | np.ndarray, length: float = 120.0, width: float = 80.0
) -> tuple[float, float]:
    """Recover StatsBomb pitch coordinates from the stored node features."""
    x = (float(node[0]) + 1.0) * length / 2.0
    y = (float(node[1]) + 1.0) * width / 2.0
    return x, y


def _actor_location(row: dict[str, Any]) -> tuple[float, float]:
    nodes = row["nodes"]
    for node in nodes:
        if float(node[13]) > 0.5:
            return _denormalise_node_location(node)
    return _denormalise_node_location(nodes[0])


def _xt_training_actions(rows: list[dict[str, Any]]) -> list[XTAction]:
    actions: list[XTAction] = []
    for row in rows:
        endpoint = row.get("endpoint")
        if endpoint is None:
            continue
        actions.append(
            XTAction(
                action_type="pass",
                start=_actor_location(row),
                end=(float(endpoint[0]), float(endpoint[1])),
                success=bool(row["pass_completed"]),
                goal=bool(row.get("goal_in_possession", 0)),
            )
        )
    return actions


def _softmax_probabilities(
    scores_by_event: list[np.ndarray], candidates_by_event: list[np.ndarray], max_nodes: int
) -> np.ndarray:
    probability = np.zeros((len(candidates_by_event), max_nodes), dtype=float)
    for event, (scores, candidates) in enumerate(zip(scores_by_event, candidates_by_event)):
        if len(candidates) == 0:
            continue
        valid_scores = scores.astype(float)
        weights = np.exp(valid_scores - valid_scores.max())
        probability[event, candidates] = weights / weights.sum()
    return probability


def _xt_receiver_probabilities(
    model: ExpectedThreatGrid, rows: list[dict[str, Any]], max_nodes: int
) -> np.ndarray:
    scores_by_event: list[np.ndarray] = []
    candidates_by_event: list[np.ndarray] = []
    for row in rows:
        candidates = np.flatnonzero(np.asarray(row["candidate_mask"][:max_nodes], dtype=bool))
        candidates_by_event.append(candidates)
        start = _actor_location(row)
        nodes = np.asarray(row["nodes"][:max_nodes], dtype=float)
        scores = np.asarray(
            [model.delta(start, _denormalise_node_location(nodes[int(candidate)])) for candidate in candidates],
            dtype=float,
        )
        scores_by_event.append(scores)
    return _softmax_probabilities(scores_by_event, candidates_by_event, max_nodes)


def _xt_value_predictions(model: ExpectedThreatGrid, rows: list[dict[str, Any]]) -> np.ndarray:
    predictions = []
    for row in rows:
        endpoint = row.get("endpoint")
        if endpoint is None:
            predictions.append(0.0)
        else:
            predictions.append(model.delta(_actor_location(row), (float(endpoint[0]), float(endpoint[1]))))
    return np.asarray(predictions, dtype=float)


def _vaep_availability() -> dict[str, Any]:
    try:
        SoccerActionVAEPAdapter()
    except RuntimeError as error:
        return {
            "status": "not_available",
            "reason": str(error),
            "interpretation": (
                "VAEP is treated as an optional socceraction-dependent baseline and is not "
                "silently replaced with a proxy."
            ),
        }
    return {
        "status": "available_not_run",
        "reason": (
            "socceraction is importable, but the current public 360 preprocessing does not yet "
            "export full SPADL action sequences required for a manuscript-grade VAEP comparison."
        ),
    }


def _chosen_data(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray([row["nodes"][int(row["receiver_index"])] for row in rows], dtype=float)
    completed = np.asarray([row["pass_completed"] for row in rows], dtype=int)
    value = np.asarray([row["value_delta_proxy"] for row in rows], dtype=float)
    return features, completed, value


def evaluate_baselines(config: dict[str, Any], fold: str) -> dict[str, Any]:
    max_nodes = int(config["model"]["max_nodes"])
    seed = int(config["project"]["seed"])
    train_rows = _dataset(config, fold, "train").rows
    test_rows = _dataset(config, fold, "test").rows
    train_x, train_selected, _ = _choice_data(train_rows, max_nodes)
    test_x, _, test_candidates = _choice_data(test_rows, max_nodes)
    labels = np.asarray([row["receiver_index"] for row in test_rows], dtype=int)

    receiver: dict[str, Any] = {
        "progressive_heuristic": receiver_metrics(
            _heuristic_probabilities(test_rows, max_nodes), labels
        )
    }
    receiver_probability: dict[str, np.ndarray] = {}
    xt_model = ExpectedThreatGrid().fit(_xt_training_actions(train_rows))
    xt_receiver_probability = _xt_receiver_probabilities(xt_model, test_rows, max_nodes)
    receiver["grid_xt"] = receiver_metrics(xt_receiver_probability, labels)
    for name, nonlinear in [("logistic_regression", False), ("hist_gradient_boosting", True)]:
        model = CandidateChoiceBaseline(nonlinear=nonlinear, random_state=seed).fit(
            train_x, train_selected
        )
        probability = _event_probabilities(model.predict_score(test_x), test_candidates, max_nodes)
        receiver_probability[name] = probability
        receiver[name] = receiver_metrics(probability, labels)

    chosen_train_x, train_completed, train_value = _chosen_data(train_rows)
    chosen_test_x, test_completed, test_value = _chosen_data(test_rows)
    completion_models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_depth=6, random_state=seed
        ),
    }
    completion = {}
    completion_probability: dict[str, np.ndarray] = {}
    for name, model in completion_models.items():
        model.fit(chosen_train_x, train_completed)
        probability = model.predict_proba(chosen_test_x)[:, 1]
        completion_probability[name] = probability
        completion[name] = binary_metrics(test_completed, probability)

    value_models = {
        "ridge": Ridge(alpha=1.0),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, max_depth=6, random_state=seed
        ),
    }
    value = {}
    value_prediction: dict[str, np.ndarray] = {}
    xt_prediction = _xt_value_predictions(xt_model, test_rows)
    value["grid_xt"] = value_metrics(test_value, xt_prediction)
    for name, model in value_models.items():
        model.fit(chosen_train_x, train_value)
        prediction = model.predict(chosen_test_x)
        value_prediction[name] = prediction
        value[name] = value_metrics(test_value, prediction)

    run_name = artifact_run_name(config)
    neural_path = Path("artifacts/results") / run_name / fold / "predictions.jsonl"
    paired_differences = None
    if neural_path.exists():
        with neural_path.open("r", encoding="utf-8") as handle:
            neural_rows = [json.loads(line) for line in handle]
        expected_ids = [str(row["event_id"]) for row in test_rows]
        observed_ids = [str(row["event_id"]) for row in neural_rows]
        if expected_ids != observed_ids:
            raise ValueError("Neural and baseline test-event order does not match")
        groups = np.asarray([str(row["match_id"]) for row in test_rows])
        neural_receiver = np.asarray(
            [row["receiver_probability"] for row in neural_rows], dtype=float
        )
        neural_receiver /= neural_receiver.sum(axis=1, keepdims=True)
        neural_completion = np.asarray(
            [row["completion_probability"][int(row["receiver_index"])] for row in neural_rows],
            dtype=float,
        )
        neural_value = np.asarray(
            [row["value_mean"][int(row["receiver_index"])] for row in neural_rows], dtype=float
        )
        repetitions = int(config.get("statistics", {}).get("cluster_bootstrap_repetitions", 1000))
        paired_differences = {
            "receiver": paired_cluster_bootstrap_differences(
                groups=groups,
                labels=labels,
                model_prediction=neural_receiver,
                baseline_prediction=receiver_probability["hist_gradient_boosting"],
                metric_function=lambda y, p: receiver_metrics(p, y),
                repetitions=repetitions,
                seed=seed,
            ),
            "completion": paired_cluster_bootstrap_differences(
                groups=groups,
                labels=test_completed,
                model_prediction=neural_completion,
                baseline_prediction=completion_probability["hist_gradient_boosting"],
                metric_function=binary_metrics,
                repetitions=repetitions,
                seed=seed + 1,
            ),
            "value": paired_cluster_bootstrap_differences(
                groups=groups,
                labels=test_value,
                model_prediction=neural_value,
                baseline_prediction=value_prediction["hist_gradient_boosting"],
                metric_function=value_metrics,
                repetitions=repetitions,
                seed=seed + 2,
            ),
        }

    report = {
        "fold": fold,
        "train_events": len(train_rows),
        "test_events": len(test_rows),
        "receiver": receiver,
        "completion": completion,
        "value": value,
        "vaep": _vaep_availability(),
        "pluralpass_minus_hist_gradient_boosting": paired_differences,
    }
    output = Path("artifacts/baselines") / run_name / fold
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        event_id=np.asarray([row["event_id"] for row in test_rows]),
        receiver_hist_gradient_boosting=receiver_probability["hist_gradient_boosting"],
        completion_hist_gradient_boosting=completion_probability["hist_gradient_boosting"],
        value_hist_gradient_boosting=value_prediction["hist_gradient_boosting"],
    )
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
