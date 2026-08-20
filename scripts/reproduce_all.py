#!/usr/bin/env python3
"""Run the ordered PluralPass football-data workflow.

This entry point never substitutes bundled example data. It either uses the
version-pinned upstream cohort or fails with the missing input clearly named.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pluralpass.config import load_config


def run_cli(config: Path, command: str, fold: str | None = None) -> None:
    invocation = [
        sys.executable,
        "-m",
        "pluralpass.cli",
        command,
        "--config",
        str(config),
    ]
    if fold is not None:
        invocation.extend(["--fold", fold])
    print("+", " ".join(invocation), flush=True)
    subprocess.run(invocation, cwd=ROOT, check=True)


def available_folds(config: dict) -> list[str]:
    manifest = Path(config["data"]["processed_dir"]) / "splits" / "manifest.json"
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not manifest.exists():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest}. Run make-splits before selecting folds."
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return sorted(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--fold", action="append", help="Fold id; may be supplied repeatedly.")
    selection.add_argument("--all-folds", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)

    if not args.skip_download:
        run_cli(config_path, "download")
    if not args.skip_preprocessing:
        run_cli(config_path, "preprocess")
        run_cli(config_path, "audit")
        run_cli(config_path, "make-splits")

    folds = available_folds(config) if args.all_folds else list(args.fold)
    unknown = sorted(set(folds) - set(available_folds(config)))
    if unknown:
        raise ValueError(f"Unknown fold(s): {unknown}")

    for fold in folds:
        if not args.skip_baselines:
            run_cli(config_path, "baselines", fold)
        if not args.skip_training:
            run_cli(config_path, "train", fold)
        if not args.skip_evaluation:
            run_cli(config_path, "evaluate", fold)

    if args.all_folds and not args.skip_evaluation:
        for command in [
            "report-formal-results",
            "set-efficiency",
            "calibration-diagnostics",
            "baseline-matrix",
        ]:
            run_cli(config_path, command)


if __name__ == "__main__":
    main()
