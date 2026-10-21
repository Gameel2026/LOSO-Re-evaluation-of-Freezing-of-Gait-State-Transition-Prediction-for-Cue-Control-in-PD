# Post hoc analysis: chance reference for coverage, episode durations and false-alarm time base

Status: post hoc. Written after the seed-42 external results and the earlier post hoc analyses were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. No model is retrained and no operating point is selected.

## Purpose

1. Coverage in the external cohorts relative to chance. The circular-shift reference of the registered analysis was applied to timing metrics only. The same reference is applied here, unchanged, to coverage (episodes with at least one cue-on window) for the registered cue states of the primary rule and controller M in tdcsfog and defog. No expectation is stated; the result is reported as obtained.
2. Episode durations in all five cohorts (median, quartiles, share of episodes lasting at most 1 s, 2 s and 5 s), as context for coverage and latency.
3. False alarms per hour on two time bases in all five cohorts: all analysed hours (as registered) and hours outside the episode zone [onset − 4 s, offset + 3 s].

## Procedure

Inputs: held-out per-window files at seed 42 (`results_mx2/MX2_per_window_<dataset>_seed42.csv.gz`, `results_mx2_external/EV_per_window_<cohort>_seed42.csv.gz`). Episodes, segments, the episode zone and false alarms as defined in `mx_new_controllers.Ev`. Circular shift within each segment, 200 shifts, random generator seeded with 42 for each controller and cohort, one-sided p = (1 + number of shifts with coverage at least as high as observed) / 201.

Script: `mx2_final_descriptives.py`. Outputs: `results_final_descriptives/FD_T1_episode_durations.csv`, `FD_T2_false_alarms_time_base.csv`, `FD_T3_coverage_surrogate.csv`.
