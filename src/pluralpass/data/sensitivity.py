from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

INTERNATIONAL_COMPETITIONS = {"FIFA World Cup", "UEFA Euro"}
CLUB_COMPETITIONS = {"1. Bundesliga", "La Liga", "Ligue 1", "Major League Soccer"}


@dataclass(frozen=True)
class CohortDefinition:
    key: str
    description: str
    rationale: str


COHORTS = [
    CohortDefinition(
        key="public_mens_360_census",
        description="All eligible men's matches with public StatsBomb event and 360 files.",
        rationale=(
            "Primary cohort. This is a version-pinned census of available public men's "
            "360 data, not a discretionary sample of competitions."
        ),
    ),
    CohortDefinition(
        key="international_tournaments",
        description="FIFA World Cup and UEFA European Championship domains only.",
        rationale=(
            "Tests whether conclusions hold in national-team tournaments where match "
            "importance, squad assembly and tactical preparation differ from club leagues."
        ),
    ),
    CohortDefinition(
        key="club_leagues",
        description="Domestic club-league domains only.",
        rationale=(
            "Tests whether conclusions hold when the cohort is restricted to recurring "
            "league environments rather than knockout or tournament football."
        ),
    ),
    CohortDefinition(
        key="exclude_small_domains",
        description="Primary cohort excluding domains below the configured pass-count threshold.",
        rationale=(
            "Checks that conclusions are not driven by very small public-data domains such "
            "as a competition-season represented by only a few matches."
        ),
    ),
    CohortDefinition(
        key="domain_balanced_large_domains",
        description="Equal-sized deterministic event sample from each sufficiently large domain.",
        rationale=(
            "Separates algorithmic performance from unequal domain sizes by preventing "
            "World Cup or Euro domains from dominating aggregate estimates."
        ),
    ),
]


def _read_passes(processed: Path) -> pd.DataFrame:
    parquet = processed / "passes.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(processed / "passes.csv.gz")


def _domain_summary(data: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        data.groupby("domain", sort=True)
        .agg(
            passes=("event_id", "count"),
            matches=("match_id", "nunique"),
            completion_rate=("pass_completed", "mean"),
            median_visible_players=("visible_players", "median"),
            median_visible_area_fraction=("visible_area_fraction", "median"),
        )
        .reset_index()
    )
    return grouped.to_dict(orient="records")


def _write_cohort(path: Path, cohort: pd.DataFrame) -> None:
    output = cohort[["event_id", "match_id", "domain", "competition", "season"]].copy()
    output.to_csv(path, index=False)


def build_sensitivity_cohorts(config: dict[str, Any]) -> dict[str, Any]:
    """Create auditable cohort definitions for data-source sensitivity analyses."""

    processed = Path(config["data"]["processed_dir"])
    data = _read_passes(processed)
    output_dir = processed / "sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["project"]["seed"])
    settings = config.get("sensitivity", {})
    minimum_domain_passes = int(settings.get("minimum_domain_passes", 5000))
    domain_balance_cap = int(settings.get("domain_balance_cap", 8000))

    large_domains = (
        data.groupby("domain")["event_id"]
        .count()
        .loc[lambda series: series >= minimum_domain_passes]
    )
    large_domain_names = set(large_domains.index.tolist())

    cohort_frames: dict[str, pd.DataFrame] = {
        "public_mens_360_census": data,
        "international_tournaments": data[
            data["competition"].isin(INTERNATIONAL_COMPETITIONS)
        ].copy(),
        "club_leagues": data[data["competition"].isin(CLUB_COMPETITIONS)].copy(),
        "exclude_small_domains": data[data["domain"].isin(large_domain_names)].copy(),
    }

    balanced_pieces = []
    for domain in sorted(large_domain_names):
        domain_frame = data[data["domain"] == domain]
        sample_size = min(domain_balance_cap, len(domain_frame))
        domain_seed = int(hashlib.sha256(domain.encode("utf-8")).hexdigest()[:8], 16)
        balanced_pieces.append(domain_frame.sample(n=sample_size, random_state=seed + domain_seed))
    cohort_frames["domain_balanced_large_domains"] = pd.concat(
        balanced_pieces, ignore_index=True
    ).sort_values(["domain", "match_id", "event_id"])

    definitions = {cohort.key: cohort for cohort in COHORTS}
    manifest: dict[str, Any] = {
        "purpose": (
            "Sensitivity cohorts addressing whether the heterogeneous public-data composition "
            "affects PluralPass conclusions."
        ),
        "primary_sampling_statement": (
            "The primary cohort is the version-pinned census of public StatsBomb men's matches "
            "with both event and 360 data at the recorded repository commit. Competitions and "
            "seasons were therefore determined by public data availability, not by post-hoc "
            "selection for model performance."
        ),
        "settings": {
            "minimum_domain_passes": minimum_domain_passes,
            "domain_balance_cap": domain_balance_cap,
            "seed": seed,
        },
        "cohorts": {},
        "domain_summary": _domain_summary(data),
    }

    for key, frame in cohort_frames.items():
        csv_path = output_dir / f"{key}.csv"
        _write_cohort(csv_path, frame)
        definition = definitions[key]
        manifest["cohorts"][key] = {
            "description": definition.description,
            "rationale": definition.rationale,
            "events": len(frame),
            "matches": int(frame["match_id"].nunique()),
            "domains": int(frame["domain"].nunique()),
            "domain_names": sorted(frame["domain"].unique().tolist()),
            "path": str(csv_path),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_markdown_report(output_dir / "README.md", manifest)
    return manifest


def _write_markdown_report(path: Path, manifest: dict[str, Any]) -> None:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Sensitivity cohorts for heterogeneous football data",
        "",
        manifest["primary_sampling_statement"],
        "",
        "These cohorts are designed to answer the likely reviewer concern that the main",
        "dataset mixes competition types, leagues and seasons. The main analysis should",
        "report leave-one-domain-out results; these cohorts support supplementary checks",
        "that restrict or rebalance the available public data.",
        "",
        "| Cohort | Events | Matches | Domains | Purpose |",
        "|---|---:|---:|---:|---|",
    ]
    for key, cohort in manifest["cohorts"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(cohort["events"]),
                    str(cohort["matches"]),
                    str(cohort["domains"]),
                    cell(cohort["rationale"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Domain summary",
            "",
            "| Domain | Passes | Matches | Completion rate | Median visible players |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for domain in manifest["domain_summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    cell(domain["domain"]),
                    str(int(domain["passes"])),
                    str(int(domain["matches"])),
                    f"{float(domain['completion_rate']):.3f}",
                    f"{float(domain['median_visible_players']):.1f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
