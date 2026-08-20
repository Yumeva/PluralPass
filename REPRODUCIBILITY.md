# Reproducibility guide

## Scope

The public workflow reconstructs the football-data cohort, visible-candidate graphs, external-domain splits, trained models, conformal sets and model-side analyses. Human-study analyses additionally require a controlled de-identified response export that is not included here.

## Environment

- Python 3.12 or newer
- R 4.4 or newer only for crossed mixed-effects models
- GPU optional; CUDA or Apple MPS is selected when available, otherwise CPU is used

Install the Python package from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

## Fixed inputs

- StatsBomb repository: `statsbomb/open-data`
- StatsBomb commit: `b0bc9f22dd77c206ddedc1d742893b3bbe64baec`
- Project seed: `20260623`
- Main configuration: `configs/base.yaml`

Do not replace the fixed upstream commit with a moving branch when reproducing manuscript results.

## Ordered commands

```bash
pluralpass download --config configs/base.yaml
pluralpass preprocess --config configs/base.yaml
pluralpass audit --config configs/base.yaml
pluralpass make-splits --config configs/base.yaml
```

The split manifest lists the legal fold identifiers. For one example fold:

```bash
pluralpass baselines --config configs/base.yaml --fold fifa_world_cup_2022
pluralpass train --config configs/base.yaml --fold fifa_world_cup_2022
pluralpass evaluate --config configs/base.yaml --fold fifa_world_cup_2022
```

After all external-domain folds are complete:

```bash
pluralpass report-formal-results --config configs/base.yaml
pluralpass set-efficiency --config configs/base.yaml
pluralpass calibration-diagnostics --config configs/base.yaml
pluralpass baseline-matrix --config configs/base.yaml
```

`scripts/reproduce_all.py` executes the same order and can run one fold or every fold in the generated manifest.

## Secondary analyses

- `scripts/rebuild_candidate_probability_table_external.py` reconstructs scenario-level candidate scores from the correct held-out fold.
- `scripts/rebuild_matched_size_analysis.py` compares APS/RAPS with matched-size Top-k, HGB, candidate-wise MLP, fixed Top-k and exact random-set expectations.
- `scripts/run_human_models_from_clean_sources.R` fits the crossed mixed-effects models from controlled de-identified inputs.

These scripts require explicit input paths. They do not search the author's desktop, silently substitute template data or reinterpret obsolete worksheet labels.

## Expected provenance files

- `data/manifest.json`: upstream commit and file-level hashes
- `data/processed/preprocess_audit.json`: exclusion flow and cohort counts
- `data/processed/splits/manifest.json`: leave-one-domain partitions
- `artifacts/checkpoints/*/*.metadata.json`: config hash, fold, seed and device
- `artifacts/results/*/*/metrics.json`: fold metrics and cluster-bootstrap intervals

## Known boundary

Exact numerical equality across hardware is not guaranteed for all GPU kernels. Seeds, configurations and data partitions are fixed; report software and hardware versions when comparing reruns.

