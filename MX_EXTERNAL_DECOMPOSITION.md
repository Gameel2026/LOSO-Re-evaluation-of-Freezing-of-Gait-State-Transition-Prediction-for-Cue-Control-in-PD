# Post hoc descriptive analysis: decomposition of the controller-M difference, reference strategies and per-patient results (external cohorts)

Status: post hoc, descriptive. Written after the seed-42 results of `MX_EXTERNAL_VALIDATION.md` (v1.5.0, DOI 10.5281/zenodo.23488586) and the exploratory analyses (commits d9469c8, 575675f, f7c6be9) were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. No model is retrained and no operating point is selected.

## Motivation

1. Controller M differs from the primary rule in two respects: a larger random-forest ensemble (500 instead of 100 trees) and the dominance-seeking threshold selection (with a hold set that selected 0 s in all external folds). The registered comparison gives their joint effect. The decomposition separates the two components on the held-out data.
2. The reference strategies used in the development cohorts (continuous cueing and an ideal detector with 0.5, 1 and 2 s latency) were not computed for the external cohorts; they provide the context for coverage, false alarms per hour and cue-on time there, together with the time spent in FoG.
3. The registered analysis reports pooled estimates; the per-patient distribution shows how many patients had fewer false alarms with M.

## Procedure

Inputs: `EV_per_window_<cohort>_seed42.csv.gz` (held-out scores and registered cue states) and `EV_choices_<cohort>.csv` (per-fold operating points), tdcsfog and defog. The registered cue states are first reproduced from the scores and the per-fold operating points.

- Decomposition (per fold, held-out scores): primary-rule score at the M operating points, and M score at the primary-rule operating points. Two orders are reported. Order A: threshold selection = (primary score at M points) − (primary rule); score = (M) − (primary score at M points). Order B: score = (M score at primary points) − (primary rule); threshold selection = (M) − (M score at primary points). Paired patient-level bootstrap CIs (5,000 resamples), all metrics.
- Reference strategies: continuous cueing; ideal detector with 0.5, 1 and 2 s latency (cue on from onset plus latency to offset of each annotated episode). Patient-level bootstrap CIs.
- Cohort description: patients, patients with FoG, episodes, hours, time in FoG, time in the episode zone.
- Per patient (M minus primary rule): numbers of patients with fewer, equal and more false alarms per hour, median and interquartile range; the same for coverage among patients with episodes.

Script: `mx2_external_decomposition.py`. Outputs: `results_mx2_external_decomposition/EVD_T1_reference_strategies.csv`, `EVD_T2_decomposition.csv`, `EVD_T3_cohort_description.csv`, `EVD_T4_per_patient.csv`.
