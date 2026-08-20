# Figure contracts

Backend is exclusively Python/matplotlib. Main figures target a 183 mm double-column width, editable SVG/PDF text and 600 dpi TIFF export.

## Figure 1 — PluralPass overview

- **Core conclusion:** PluralPass converts a partially observed football state into calibrated, set-valued passing recommendations and evaluates them outside the development domain and against coach disagreement.
- **Archetype:** schematic-led composite.
- **Panel map:** a, partial 360 state; b, masked graph; c, separate choice/risk/value heads; d, conformal/Pareto action set; e, external and human validation.
- **Reviewer risk:** the schematic must not imply access to off-screen players or causal knowledge of unplayed passes.

## Figure 2 — Benchmark and predictive decomposition

- **Core conclusion:** PluralPass external validation is supported by a version-pinned heterogeneous public 360 benchmark, with separable receiver-choice, execution-risk and value signals and consistently positive paired gains over the strongest non-neural baseline.
- **Archetype:** quantitative grid.
- **Panel map:** a, domain composition and eligible passes; b, external-test receiver Top-1/Top-3, completion AUROC and value Spearman; c, APS coverage against the 90% target with set size and abstention; d, paired gains over histogram-gradient boosting.
- **Reviewer risk:** the figure frames the cohort as a public-data availability census and external-validation benchmark, not as a statistically representative sample of all professional football.

## Figure 3 — Calibration and external validity

- **Core conclusion:** PluralPass is well behaved under pooled reliability checks and uncertainty stratification, but its set-valued guarantee is bounded by visible receiver availability rather than by generic claims of real-time omniscience.
- **Archetype:** quantitative grid.
- **Panel map:** a, pooled receiver and completion reliability curves; b, epistemic low/mid/high strata for Top-1, set coverage and set size; c, counterfactual missingness stress tests; d, event-random match leakage and non-uniform optimism audit.
- **Reviewer risk:** the panel must emphasize method boundaries: hidden true receivers break coverage, and event-random leakage is a validation-design problem even when it does not uniformly inflate every metric.

## Figure 4 — Tactical ambiguity

- **Core conclusion:** PluralPass uncertainty corresponds to scenario-level variation in coach first-choice passing decisions, while the evidence remains correlational rather than causal.
- **Archetype:** quantitative grid.
- **Panel map:** a, model receiver entropy versus coach first-choice entropy; b, conformal set size versus coach first-choice entropy; c, locked example scenes ranging from consensus to plural first-choice distributions; d, H3 endpoint forest with scenario-cluster bootstrap intervals.
- **Reviewer risk:** avoid causal language; the figure supports correspondence with tactical ambiguity in the 40-scenario coach sample, not an explanation of individual coach cognition.

## Figure 5 — Human evaluation and cases

- **Core conclusion:** calibrated sets cover coach-endorsed options more often than point recommendations and are rated more reasonable and informative, but they carry presentation costs through lower clarity, longer decision time and more bad-option inclusions.
- **Archetype:** quantitative grid.
- **Panel map:** a, coverage of first-choice, any top-three and acceptable-set options; b, bootstrap difference forest for 1–7 assistance ratings; c, missing-option, bad-option and change-rate flags; d, stage-2 decision time by condition.
- **Reviewer risk:** report both benefits and costs; do not present set-valued assistance as uniformly superior on every usability dimension.
