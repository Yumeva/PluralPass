from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".json":
        config = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml

            with path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
        except ImportError:
            fallback = path.with_suffix(".json")
            if not fallback.exists():
                raise RuntimeError("PyYAML is unavailable and no adjacent JSON config exists")
            config = json.loads(fallback.read_text(encoding="utf-8"))
    config["_config_path"] = str(path.resolve())
    config["_config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def artifact_run_name(config: dict[str, Any]) -> str:
    """Return a filesystem-safe namespace separating formal and pilot runs."""
    name = str(config["project"]["name"])
    safe = "".join(character.lower() if character.isalnum() else "-" for character in name)
    return "-".join(part for part in safe.split("-") if part)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_run_metadata(path: str | Path, config: dict[str, Any], **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": config.get("_config_path"),
        "config_sha256": config.get("_config_sha256"),
        "seed": config["project"]["seed"],
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
