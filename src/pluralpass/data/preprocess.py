from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


from pluralpass.geometry import (
    euclidean,
    line_blockage,
    nearest_opponent_distance,
    normalise_location,
    polygon_area,
    progressive_value,
)


def _timestamp_seconds(event: dict[str, Any]) -> float:
    timestamp = event.get("timestamp", "00:00:00.000")
    hours, minutes, seconds = timestamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _is_open_play_pass(event: dict[str, Any], config: dict[str, Any]) -> bool:
    if event.get("type", {}).get("name") != "Pass":
        return False
    play_pattern = event.get("play_pattern", {}).get("name")
    if play_pattern not in set(config["data"]["open_play_patterns"]):
        return False
    pass_type = event.get("pass", {}).get("type", {}).get("name")
    return pass_type not in set(config["data"]["excluded_pass_types"])


def _future_labels(
    events: list[dict[str, Any]], index: int, horizon: float
) -> tuple[int, int, float]:
    event = events[index]
    possession = event.get("possession")
    team_id = event.get("team", {}).get("id")
    t0 = _timestamp_seconds(event)
    current = progressive_value(event.get("location", [0.0, 40.0]))
    best = current
    shot_10s = 0
    goal_possession = 0
    for subsequent in events[index + 1 :]:
        if subsequent.get("period") != event.get("period"):
            break
        if subsequent.get("possession") != possession:
            break
        dt = _timestamp_seconds(subsequent) - t0
        if (
            dt <= horizon
            and subsequent.get("team", {}).get("id") == team_id
            and subsequent.get("location")
        ):
            best = max(best, progressive_value(subsequent["location"]))
        if dt <= horizon and subsequent.get("type", {}).get("name") == "Shot":
            shot_10s = 1
        if subsequent.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            goal_possession = 1
    return shot_10s, goal_possession, float(best - current)


def _node_features(
    frame: list[dict[str, Any]], visible_area: list[float], config: dict[str, Any]
) -> list[list[float]]:
    length = config["data"]["field_length"]
    width = config["data"]["field_width"]
    actor_index = next(i for i, node in enumerate(frame) if node.get("actor"))
    actor_location = frame[actor_index]["location"]
    opponents = [node for node in frame if not node.get("teammate")]
    area_fraction = polygon_area(visible_area) / (length * width)
    features = []
    for i, node in enumerate(frame):
        x, y = normalise_location(node["location"], length, width)
        dx = (node["location"][0] - actor_location[0]) / length
        dy = (node["location"][1] - actor_location[1]) / width
        distance = euclidean(node["location"], actor_location) / length
        angle = math.atan2(dy, dx) / math.pi
        pressure = nearest_opponent_distance(i, frame) / length
        blockers = line_blockage(actor_location, node["location"], opponents) / 11.0
        goal_distance = euclidean(node["location"], (length, width / 2.0)) / length
        features.append(
            [
                x,
                y,
                float(node.get("teammate", False)),
                float(node.get("actor", False)),
                float(node.get("keeper", False)),
                dx,
                dy,
                distance,
                angle,
                pressure,
                blockers,
                goal_distance,
                area_fraction,
                float(i == actor_index),
            ]
        )
    return features


def _match_receiver(
    frame: list[dict[str, Any]], endpoint: list[float], max_distance: float
) -> tuple[int | None, float]:
    candidates = [
        (i, euclidean(node["location"], endpoint))
        for i, node in enumerate(frame)
        if node.get("teammate") and not node.get("actor")
    ]
    if not candidates:
        return None, float("inf")
    index, distance = min(candidates, key=lambda item: item[1])
    return (index if distance <= max_distance else None), float(distance)


def preprocess(config: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(config["data"]["raw_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads((raw_dir / "selected_matches.json").read_text(encoding="utf-8"))

    graph_path = processed_dir / "passes.jsonl.gz"
    metadata: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    with gzip.open(graph_path, "wt", encoding="utf-8") as output:
        for match in tqdm(selected, desc="preprocess matches"):
            match_id = int(match["match_id"])
            event_file = raw_dir / "events" / f"{match_id}.json"
            frame_file = raw_dir / "three-sixty" / f"{match_id}.json"
            if not event_file.exists() or not frame_file.exists():
                exclude("missing_match_file")
                continue
            events = json.loads(event_file.read_text(encoding="utf-8"))
            frame_rows = json.loads(frame_file.read_text(encoding="utf-8"))
            frame_by_event = {row["event_uuid"]: row for row in frame_rows}
            domain = f"{match['competition']['competition_name']}|{match['season']['season_name']}"
            for position, event in enumerate(events):
                if not _is_open_play_pass(event, config):
                    continue
                frame_row = frame_by_event.get(event["id"])
                if not frame_row or not frame_row.get("freeze_frame"):
                    exclude("missing_360")
                    continue
                frame = frame_row["freeze_frame"]
                actors = [i for i, node in enumerate(frame) if node.get("actor")]
                if len(actors) != 1:
                    exclude("invalid_actor_count")
                    continue
                candidate_mask = [
                    bool(node.get("teammate") and not node.get("actor")) for node in frame
                ]
                if sum(candidate_mask) < config["data"]["min_visible_teammates"]:
                    exclude("too_few_candidates")
                    continue
                endpoint = event.get("pass", {}).get("end_location")
                if not endpoint:
                    exclude("missing_endpoint")
                    continue
                receiver_index, receiver_distance = _match_receiver(
                    frame, endpoint, config["data"]["receiver_match_max_m"]
                )
                if receiver_index is None:
                    exclude("receiver_not_reliably_visible")
                    continue

                outcome = event.get("pass", {}).get("outcome", {}).get("name")
                completed = int(outcome is None)
                shot_10s, goal_possession, value_delta = _future_labels(
                    events, position, config["data"]["short_horizon_seconds"]
                )
                row = {
                    "event_id": event["id"],
                    "match_id": match_id,
                    "domain": domain,
                    "competition": match["competition"]["competition_name"],
                    "season": match["season"]["season_name"],
                    "minute": event.get("minute", 0),
                    "second": event.get("second", 0),
                    "score_context": None,
                    "nodes": _node_features(frame, frame_row.get("visible_area", []), config),
                    "node_mask": [True] * len(frame),
                    "candidate_mask": candidate_mask,
                    "receiver_index": receiver_index,
                    "receiver_match_distance_m": receiver_distance,
                    "pass_completed": completed,
                    "shot_within_10s": shot_10s,
                    "goal_in_possession": goal_possession,
                    "value_delta_proxy": value_delta,
                    "visible_players": len(frame),
                    "visible_area_fraction": polygon_area(frame_row.get("visible_area", []))
                    / (config["data"]["field_length"] * config["data"]["field_width"]),
                    "pass_length": event.get("pass", {}).get("length"),
                    "pass_angle": event.get("pass", {}).get("angle"),
                    "endpoint": endpoint,
                }
                output.write(json.dumps(row, separators=(",", ":")) + "\n")
                metadata.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"nodes", "node_mask", "candidate_mask"}
                    }
                )

    frame = pd.DataFrame(metadata)
    frame.to_csv(processed_dir / "passes.csv.gz", index=False, compression="gzip")
    try:
        frame.to_parquet(processed_dir / "passes.parquet", index=False)
    except (ImportError, ModuleNotFoundError):
        pass
    audit = {
        "eligible_passes": len(metadata),
        "domains": frame["domain"].value_counts().to_dict() if len(frame) else {},
        "matches": int(frame["match_id"].nunique()) if len(frame) else 0,
        "receiver_match_distance_quantiles_m": (
            frame["receiver_match_distance_m"].quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
            if len(frame)
            else {}
        ),
        "exclusions": exclusions,
    }
    (processed_dir / "preprocess_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit
