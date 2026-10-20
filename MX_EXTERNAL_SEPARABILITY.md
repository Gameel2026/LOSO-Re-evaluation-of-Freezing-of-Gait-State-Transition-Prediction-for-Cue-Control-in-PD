# Post hoc analysis: pre-onset separability in the development and external cohorts

Status: post hoc and exploratory. Written after the seed-42 results of `MX_EXTERNAL_VALIDATION.md` (v1.5.0, DOI 10.5281/zenodo.23488586; commit 4eb4d30) were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. The analysis is reported in the Supplementary material and, as one sentence, in Method validation.

## Motivation

Pre-onset separability was used in the development cohorts as a pre-check of whether anticipatory cue activation can be expected. In the external cohorts, timely activation of controller M exceeded the circular-shift reference in the laboratory cohort (tdcsfog) and did not exceed it in the home cohort (defog). The analysis measures pre-onset separability with one identical procedure in all five cohorts, so that the external timing results can be related to separability measured in the same cohorts rather than in the development data.

## Expectation stated before execution

If the pre-check is informative, pre-onset separability (lead of at least 1 s) is lower in defog than in tdcsfog. All results are reported whatever their direction.

## Procedure

The held-out (leave-one-subject-out) per-window scores of controller M at seed 42 are used without retraining: `results_mx2/MX2_per_window_<dataset>_seed42.csv.gz` for DAPHNET, Multimodal FoG and FoG-STAR, and `results_mx2_external/EV_per_window_<cohort>_seed42.csv.gz` for tdcsfog and defog.

Window sets, with the pre-onset zone of 8 windows and the post-offset zone of 6 windows used throughout (`mx_new_controllers.PRE_W`, `POST_W`):

- No-FoG: windows outside every episode zone [onset − 8, offset + 6] and not FoG.
- Pre-onset, lead of at least 1 s: windows 2 to 8 before each annotated onset, within the same recording segment, not FoG, and not within the post-offset zone of a preceding episode.
- Last window before onset: the window immediately before each onset, under the same conditions.
- FoG (reference): annotated FoG windows.

For each cohort and each positive set against No-FoG: pooled AUC of the controller-M score, patient-level bootstrap 95% CI (5,000 resamples of patients, computed exactly from per-patient pair counts), and the median of per-patient AUCs (patients with both classes). The difference in pooled AUC (lead of at least 1 s) between tdcsfog and defog is reported with a bootstrap CI from independent resamples of the two cohorts.

## Scope of interpretation

Separability here is a property of the controller score, measured on held-out patients. It is not used to select any parameter and it does not restate EV1 to EV4.

Script: `mx2_external_separability.py`. Outputs: `results_mx2_external_separability/EVSEP_T1_separability.csv`, `EVSEP_T2_external_difference.csv`.
