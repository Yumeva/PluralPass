from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import yaml

from pluralpass import __version__

ROOT = Path(__file__).resolve().parents[1]
IGNORED_GENERATED_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
}
FORBIDDEN_MARKERS = (
    "/" + "Users/",
    "b0bc9f22dd77c206ddedc1d7428933" + "bbe64baec",
    "[" + "To be added after release]",
    "simulated coach",
    "synthetic coach",
    "simulated judgement",
    "simulated judgment",
    "synthetic ratings",
    "synthetic_realism",
    "coach_analysis_uploaded",
    "PluralPass_coach_uploaded_analysis",
)


def test_public_text_and_code_have_no_machine_paths_or_obsolete_labels() -> None:
    suffixes = {".py", ".R", ".md", ".toml", ".yaml", ".yml", ".json", ".csv", ".cff"}
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path == Path(__file__)
            or path.suffix not in suffixes
            or any(part in IGNORED_GENERATED_DIRECTORIES for part in path.parts)
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        normalized_text = text.lower()
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in normalized_text:
                failures.append(f"{path.relative_to(ROOT)} contains {marker!r}")
    assert not failures, "\n".join(failures)


def test_public_release_contains_no_participant_workbooks_or_response_exports() -> None:
    restricted_suffixes = {".xlsx", ".xls", ".sav", ".dta"}
    restricted_names = ("coach_response", "participant_response", "consent", "identity_key")
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(
            part in IGNORED_GENERATED_DIRECTORIES for part in path.parts
        ):
            continue
        relative = path.relative_to(ROOT)
        lowered = str(relative).lower()
        if path.suffix.lower() in restricted_suffixes or any(
            marker in lowered for marker in restricted_names
        ):
            failures.append(str(relative))
    assert not failures, "Restricted human-study artifacts in public tree:\n" + "\n".join(failures)


def test_dataset_metadata_matches_feature_dictionary() -> None:
    metadata = json.loads(
        (ROOT / "data" / "metadata" / "dataset_info.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in (ROOT / "data" / "metadata" / "feature_description.csv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.strip()
    ]
    assert metadata["node_feature_count"] == 14
    assert len(rows) == 15


def test_figure_modules_force_a_headless_backend() -> None:
    env = os.environ.copy()
    env["MPLBACKEND"] = "MacOSX"
    command = [
        sys.executable,
        "-c",
        (
            "import matplotlib; import figures.make_figure1; "
            "assert matplotlib.get_backend().lower() == 'agg', matplotlib.get_backend()"
        ),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_release_versions_are_consistent() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    requirements = {
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert project["project"]["version"] == __version__
    assert citation["version"] == __version__
    assert set(project["project"]["dependencies"]) == requirements
