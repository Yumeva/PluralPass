from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from pluralpass import __version__

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        return iterable


GITHUB_API = "https://api.github.com"
RAW_GITHUB = "https://raw.githubusercontent.com"


def _get_json(url: str, timeout: int = 90) -> Any:
    response = requests.get(
        url,
        headers={"User-Agent": f"PluralPass/{__version__}"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _download(url: str, destination: Path, attempts: int = 5) -> dict[str, Any]:
    if destination.exists() and destination.stat().st_size > 0:
        content = destination.read_bytes()
        return {
            "path": str(destination),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
            "reused": True,
        }
    content: bytes | None = None
    last_error: requests.RequestException | None = None
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers={"User-Agent": f"PluralPass/{__version__}"},
                timeout=180,
            )
            response.raise_for_status()
            content = response.content
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    if content is None:
        raise RuntimeError(f"failed to download {url} after {attempts} attempts") from last_error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return {
        "path": str(destination),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def download_open_360(config: dict[str, Any], workers: int = 32) -> dict[str, Any]:
    data_cfg = config["data"]
    repo = data_cfg["upstream_repo"]
    ref = data_cfg["upstream_ref"]
    raw_dir = Path(data_cfg["raw_dir"])
    manifest_path = Path(data_cfg["manifest_path"])

    commit = _get_json(f"{GITHUB_API}/repos/{repo}/commits/{ref}")["sha"]
    tree = _get_json(f"{GITHUB_API}/repos/{repo}/git/trees/{commit}?recursive=1")
    paths = [entry["path"] for entry in tree["tree"] if entry["type"] == "blob"]
    match_indexes = [p for p in paths if p.startswith("data/matches/") and p.endswith(".json")]
    three_sixty_paths = [
        p for p in paths if p.startswith("data/three-sixty/") and p.endswith(".json")
    ]
    ids_360 = {Path(path).stem for path in three_sixty_paths}

    def raw_url(path: str) -> str:
        return f"{RAW_GITHUB}/{repo}/{commit}/{path}"

    competitions = _get_json(raw_url("data/competitions.json"))
    gender_by_domain = {
        (int(row["competition_id"]), int(row["season_id"])): str(row.get("competition_gender", ""))
        for row in competitions
    }

    index_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_get_json, raw_url(path)) for path in match_indexes]
        for future in tqdm(as_completed(futures), total=len(futures), desc="match indexes"):
            index_rows.extend(future.result())

    selected = [
        row
        for row in index_rows
        if str(row["match_id"]) in ids_360
        and gender_by_domain.get(
            (int(row["competition"]["competition_id"]), int(row["season"]["season_id"])), ""
        ).lower()
        == "male"
    ]
    selected_ids = {str(row["match_id"]) for row in selected}

    # Persist the selection before downloading the much larger event payloads.
    # This makes interrupted downloads resumable and lets alternative bulk
    # transports reuse the exact same, version-pinned cohort.
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "selected_matches.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    required = ["data/competitions.json"]
    for match_id in sorted(selected_ids):
        required.extend(
            [
                f"data/events/{match_id}.json",
                f"data/three-sixty/{match_id}.json",
            ]
        )

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download, raw_url(path), raw_dir / path.removeprefix("data/")): path
            for path in required
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="files"):
            records.append(future.result())

    domains: dict[str, int] = {}
    for row in selected:
        key = f"{row['competition']['competition_name']}|{row['season']['season_name']}"
        domains[key] = domains.get(key, 0) + 1

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "upstream_repo": repo,
        "upstream_commit": commit,
        "male_360_matches": len(selected),
        "domains": dict(sorted(domains.items())),
        "match_ids": sorted(int(x) for x in selected_ids),
        "files": sorted(records, key=lambda x: x["path"]),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
