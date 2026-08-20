from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def make_splits(config: dict[str, Any]) -> dict[str, Any]:
    processed = Path(config["data"]["processed_dir"])
    parquet = processed / "passes.parquet"
    data = (
        pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(processed / "passes.csv.gz")
    )
    split_dir = processed / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["seed"])
    cal_fraction = float(config["splits"]["calibration_fraction"])
    val_fraction = float(config["splits"]["validation_fraction"])
    manifests = {}

    for fold_index, held_domain in enumerate(sorted(data["domain"].unique())):
        test = data[data["domain"] == held_domain].copy()
        development = data[data["domain"] != held_domain].copy()
        first = GroupShuffleSplit(
            n_splits=1, test_size=cal_fraction + val_fraction, random_state=seed + fold_index
        )
        train_idx, remainder_idx = next(
            first.split(development, groups=development[config["splits"]["group_key"]])
        )
        train = development.iloc[train_idx].copy()
        remainder = development.iloc[remainder_idx].copy()
        relative_val = val_fraction / (cal_fraction + val_fraction)
        second = GroupShuffleSplit(
            n_splits=1, test_size=relative_val, random_state=seed + 1000 + fold_index
        )
        cal_idx, val_idx = next(
            second.split(remainder, groups=remainder[config["splits"]["group_key"]])
        )
        calibration = remainder.iloc[cal_idx].copy()
        validation = remainder.iloc[val_idx].copy()

        pieces = []
        for frame, name in [
            (train, "train"),
            (calibration, "calibration"),
            (validation, "validation"),
            (test, "test"),
        ]:
            subset = frame[["event_id", "match_id", "domain"]].copy()
            subset["split"] = name
            pieces.append(subset)
        split = pd.concat(pieces, ignore_index=True)
        assert split.groupby("match_id")["split"].nunique().max() == 1
        fold = slugify(held_domain)
        split.to_csv(split_dir / f"{fold}.csv", index=False)
        manifests[fold] = {
            "held_domain": held_domain,
            "events": split["split"].value_counts().to_dict(),
            "matches": split.groupby("split")["match_id"].nunique().to_dict(),
            "domains_in_train": sorted(train["domain"].unique().tolist()),
        }

    (split_dir / "manifest.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return manifests


def make_grouped_kfold_splits(config: dict[str, Any], n_splits: int = 10) -> dict[str, Any]:
    """Create match-grouped k-fold splits for supplementary stability analysis.

    Unlike the event-random diagnostic, a match never appears in more than one
    partition within a fold. Test folds are assigned at match level; the
    remaining matches are then split into train/calibration/validation by match.
    """

    processed = Path(config["data"]["processed_dir"])
    parquet = processed / "passes.parquet"
    data = (
        pd.read_parquet(parquet) if parquet.exists() else pd.read_csv(processed / "passes.csv.gz")
    )
    split_dir = processed / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["project"]["seed"])
    cal_fraction = float(config["splits"]["calibration_fraction"])
    val_fraction = float(config["splits"]["validation_fraction"])
    group_key = config["splits"]["group_key"]

    match_sizes = data.groupby(group_key).size().reset_index(name="events")
    match_sizes = match_sizes.sample(frac=1.0, random_state=seed).sort_values(
        ["events", group_key], ascending=[False, True]
    )
    fold_event_totals = [0 for _ in range(n_splits)]
    fold_matches: list[list[Any]] = [[] for _ in range(n_splits)]
    for row in match_sizes.itertuples(index=False):
        target = min(range(n_splits), key=lambda index: fold_event_totals[index])
        match_id = getattr(row, group_key)
        events = int(row.events)
        fold_matches[target].append(match_id)
        fold_event_totals[target] += events

    manifests: dict[str, Any] = {}
    for fold_index, test_matches in enumerate(fold_matches, start=1):
        fold = f"grouped10_fold_{fold_index:02d}"
        test_match_set = set(test_matches)
        test = data[data[group_key].isin(test_match_set)].copy()
        development = data[~data[group_key].isin(test_match_set)].copy()

        first = GroupShuffleSplit(
            n_splits=1,
            test_size=cal_fraction + val_fraction,
            random_state=seed + 5000 + fold_index,
        )
        train_idx, remainder_idx = next(first.split(development, groups=development[group_key]))
        train = development.iloc[train_idx].copy()
        remainder = development.iloc[remainder_idx].copy()

        relative_val = val_fraction / (cal_fraction + val_fraction)
        second = GroupShuffleSplit(
            n_splits=1,
            test_size=relative_val,
            random_state=seed + 6000 + fold_index,
        )
        cal_idx, val_idx = next(second.split(remainder, groups=remainder[group_key]))
        calibration = remainder.iloc[cal_idx].copy()
        validation = remainder.iloc[val_idx].copy()

        pieces = []
        for frame, name in [
            (train, "train"),
            (calibration, "calibration"),
            (validation, "validation"),
            (test, "test"),
        ]:
            subset = frame[["event_id", "match_id", "domain"]].copy()
            subset["split"] = name
            pieces.append(subset)
        split = pd.concat(pieces, ignore_index=True)
        assert split.groupby("match_id")["split"].nunique().max() == 1
        split.to_csv(split_dir / f"{fold}.csv", index=False)
        manifests[fold] = {
            "scheme": "match_grouped_10fold",
            "events": split["split"].value_counts().to_dict(),
            "matches": split.groupby("split")["match_id"].nunique().to_dict(),
            "test_domains": sorted(test["domain"].unique().tolist()),
        }

    manifest_path = split_dir / "grouped_10fold_manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    return {
        "scheme": "match_grouped_10fold",
        "n_splits": n_splits,
        "manifest_path": str(manifest_path),
        "folds": manifests,
        "interpretation": (
            "These folds are supplementary stability checks. Leave-one-domain "
            "validation remains the primary out-of-domain estimate."
        ),
    }
