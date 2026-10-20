# Post hoc sensitivity analysis: hysteresis thresholds below the pre-specified grid (external cohorts)

Status: post hoc and exploratory. Written after the seed-42 results of `MX_EXTERNAL_VALIDATION.md` (v1.5.0, DOI 10.5281/zenodo.23488586) were available, and committed together with its script before the script was executed. EV1 to EV4, their verdicts and the pre-specified threshold grid are not changed; this analysis is reported in the Supplementary material only.

## Motivation

In the home cohort (defog), the inner selection placed both the primary rule and controller M at on 0.30 and off 0.10 in all 45 folds, which is the lower bound of the pre-specified grid (on 0.30 to 0.90; off 0.10 to on). The analysis describes how the two controllers behave at and below this bound, and whether the difference between them depends on it.

## Procedure

The held-out per-window scores saved by the seed-42 analysis (`EV_per_window_<cohort>_seed42.csv.gz`) are used without retraining. The per-fold operating points of `EV_choices_<cohort>.csv` are first re-applied to these scores to confirm that the registered cue states are reproduced. Fixed operating points are then applied to all patients of a cohort: on from 0.10 to 0.90 in steps of 0.05, off in {0.05, 0.10, 0.20} with off below on, hold 0 s. No operating point is selected; the analysis is a descriptive operating curve on held-out data.

## Outputs

For each cohort and operating point: coverage, timely activation, timely activation with lead of at least 1 s, cue at onset, false alarms per hour and cue-on time for both controllers, and the paired difference (M minus primary rule) with patient-level bootstrap 95% CIs (5,000 resamples). All operating points are reported.

## Scope of interpretation

The results are not used to select thresholds or to restate EV1 to EV4. They show whether coverage and false alarms change below the grid bound and whether the false-alarm difference between M and the primary rule is preserved there.

Script: `mx2_external_grid_sensitivity.py`. Outputs: `results_mx2_external_grid/EVG_T0_reproduction.csv`, `EVG_T1_operating_points.csv`, `EVG_T2_paired_differences.csv`.
