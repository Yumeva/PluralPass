# PluralPass

Reference implementation for **PluralPass**, a set-valued prediction framework for open-play football passing under partial observability.

PluralPass represents each StatsBomb 360 freeze frame as a variable-size interaction graph, separates receiver selection from conditional completion and short-horizon spatial progression, estimates epistemic uncertainty with a deep ensemble, and converts receiver probabilities into calibrated candidate sets with APS or RAPS.

> **Release status.** This repository contains the executable research code and fixed configurations used by the project. Third-party StatsBomb files, trained checkpoints and individual-level coach responses are not redistributed. Commands that require those inputs fail explicitly rather than substituting placeholder data.

## What is included

- version-pinned acquisition of male StatsBomb Open Data matches with 360 freeze frames;
- open-play pass filtering, visible-candidate construction and the documented 14-feature node representation;
- match-grouped leave-one-domain splits and grouped 10-fold sensitivity splits;
- graph-transformer PluralPass model, candidate-wise MLP and non-neural baselines;
- deep-ensemble training and evaluation for receiver, completion and spatial-progression targets;
- APS and RAPS prediction sets, abstention rules, calibration diagnostics and stress tests;
- matched-size set comparisons and de-identified coach-analysis utilities;
- scripts for the manuscript figures when their corresponding source-data tables are available.

## Repository layout

```text
PluralPass/
├── configs/                 Fixed model, baseline and ablation configurations
├── data/                    Metadata plus ignored raw/interim/processed folders
├── figures/                 Publication-figure code and visual contracts
├── scripts/                 Reproduction and secondary-analysis entry points
├── src/pluralpass/          Installable Python package
├── tests/                   Deterministic unit and smoke tests
├── DATA_AVAILABILITY.md     Public, restricted and non-redistributed inputs
├── REPRODUCIBILITY.md       Ordered commands and expected artifacts
├── pyproject.toml           Package and dependency specification
└── CITATION.cff             Software citation metadata
```

## Data version

The primary football-data source is [StatsBomb Open Data](https://github.com/statsbomb/open-data), fixed at commit:

```text
b0bc9f22dd77c206ddedc1d742893b3bbe64baec
```

The commit is recorded in `configs/base.yaml` and `configs/base.json`. The download command resolves that exact revision and writes file-level SHA-256 hashes to `data/manifest.json`.

This repository does not redistribute StatsBomb match files. Users must obtain them from the upstream repository and comply with its current terms and attribution requirements.

## Installation

PluralPass requires Python 3.12 or newer. A clean virtual environment is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For testing and repository checks:

```bash
python -m pip install -e '.[dev]'
pytest
```

The crossed mixed-effects models used for the coach study additionally require R 4.4 or newer and the packages listed in `R_REQUIREMENTS.txt`.

## Core workflow

Run commands from the repository root. The console entry point and `python -m pluralpass.cli` are equivalent.

```bash
# 1. Download the exact upstream revision and record provenance.
pluralpass download --config configs/base.yaml

# 2. Construct eligible open-play passes and visible receiver candidates.
pluralpass preprocess --config configs/base.yaml
pluralpass audit --config configs/base.yaml

# 3. Create match-grouped leave-one-domain partitions.
pluralpass make-splits --config configs/base.yaml

# 4. Train and evaluate one external-domain fold.
pluralpass baselines --config configs/base.yaml --fold fifa_world_cup_2022
pluralpass train --config configs/base.yaml --fold fifa_world_cup_2022
pluralpass evaluate --config configs/base.yaml --fold fifa_world_cup_2022
```

To execute the ordered workflow for one fold or all folds:

```bash
python scripts/reproduce_all.py --config configs/base.yaml --fold fifa_world_cup_2022
python scripts/reproduce_all.py --config configs/base.yaml --all-folds
```

The all-fold run downloads several gigabytes of third-party data and trains multiple ensemble members; runtime depends strongly on hardware. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) before launching it.

## Outputs

Generated files are intentionally ignored by Git and written beneath:

- `data/raw/`: version-pinned upstream JSON files;
- `data/processed/`: graph records, audit tables and split manifests;
- `artifacts/checkpoints/`: trained model states and run metadata;
- `artifacts/results/`: fold-level predictions, metrics and confidence intervals;
- `artifacts/figures/`: generated PDF, SVG and PNG figures.

Each run name is derived from the configuration, and checkpoint metadata records the configuration SHA-256, fold, seed and execution device.

## Coach evaluation

The human evaluation was approved by the Guangzhou Sport University ethics committee (`2024LCLL-71`), and informed consent was obtained before participation. Individual-level response files, contact information, consent records and identifying free text are not part of this public code repository.

The study was not externally preregistered before outcome access. The released code therefore describes its primary and sensitivity analyses without representing the completed study as prospectively preregistered. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the release boundary.

## Interpretation boundary

PluralPass ranks and calibrates **visible** teammate candidates in a single 360 freeze frame. It does not infer a unique globally optimal pass, reconstruct off-screen players, model full future trajectories or estimate VAEP/EPV. The value head is a documented short-horizon spatial-progression proxy.

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff) when citing the software. The manuscript citation and archival DOI will be added only after they exist; no placeholder DOI is presented as a completed deposit.

## License

Code in this repository is released under the [MIT License](LICENSE). Third-party data remain subject to their own terms.
