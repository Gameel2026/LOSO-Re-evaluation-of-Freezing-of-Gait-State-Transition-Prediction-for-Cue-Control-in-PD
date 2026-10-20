# Post hoc analysis: chance reference for activation timing at alternative operating points (external cohorts)

Status: post hoc and exploratory. Written after the seed-42 results of `MX_EXTERNAL_VALIDATION.md` (v1.5.0, DOI 10.5281/zenodo.23488586), the grid sensitivity analysis (`MX_EXTERNAL_GRID_SENSITIVITY.md`, commit d9469c8) and the separability analysis (`MX_EXTERNAL_SEPARABILITY.md`, commit 575675f) were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. Reported in the Supplementary material only.

## Motivation

In the home cohort (defog), activation timing of controller M did not exceed the circular-shift reference, although the controller score retained pre-onset separability (pooled AUC 0.65 for windows with lead of at least 1 s, similar to 0.62 in tdcsfog). The registered operating points lay at the lower bound of the threshold grid with 14% cue-on time. The analysis examines whether timing relative to the chance reference depends on the operating point.

## Expectation stated before execution

If the home timing result reflects the operating point rather than absent pre-onset information, timely activation with lead of at least 1 s exceeds the circular-shift reference (one-sided p below 0.05) in defog at one or more of the lower fixed operating points. All results are reported whatever their direction; no operating point is selected or recommended from this analysis.

## Procedure

Held-out per-window scores of controller M at seed 42 (`EV_per_window_<cohort>_seed42.csv.gz`) are used without retraining, in tdcsfog and defog. Operating points:

- the registered per-fold operating points (cue states as saved; this row reproduces `EV_T5_surrogate.csv`);
- six fixed operating points applied to all patients, hold 0 s: on 0.20, 0.25, 0.30 with off 0.05 and 0.10.

For each operating point, the circular-shift surrogate of the registered analysis is applied unchanged (`mx_new_controllers.surrogate`, 200 shifts within each recording segment, random generator seeded with 42): observed value, surrogate mean, surrogate 95th percentile and one-sided p for timely activation with lead of at least 1 s (primary metric of this analysis), cue at onset and timely activation, together with coverage, false alarms per hour and cue-on time. Seven operating points per cohort are tested; p values are reported without adjustment and interpreted descriptively.

Script: `mx2_external_operating_surrogate.py`. Output: `results_mx2_external_operating_surrogate/EVOS_T1_operating_point_surrogate.csv`.
