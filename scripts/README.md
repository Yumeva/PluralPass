# Analysis scripts

## Main workflow

`reproduce_all.py` runs the version-pinned football-data pipeline for one or all leave-one-domain folds. It requires an explicit fold selection and does not fall back to placeholder data.

## External-fold coach-study linkage

`rebuild_candidate_probability_table_external.py` joins each selected study scenario to predictions from the model that held out the scenario's domain.

`rebuild_matched_size_analysis.py` consumes that candidate table, a controlled de-identified response workbook and the set-efficiency results. The default worksheet names are `coach_responses` and `scenario_summary`; alternate approved names must be passed explicitly with `--responses-sheet` and `--scenario-sheet`.

`run_human_models_from_clean_sources.R` fits crossed evaluator and scenario mixed-effects models from de-identified CSV inputs.

No script in this directory contains an author-specific absolute path, embeds participant identities or silently treats obsolete template labels as data provenance.
