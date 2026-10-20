# Post hoc descriptive reporting: activation latency and composition of early activation

Status: post hoc, descriptive. Written after the seed-42 external results and the exploratory analyses were available, and committed together with its script before the script was executed. No hypothesis is tested, no model is retrained and no operating point is selected. EV1 to EV4 and their verdicts are not changed.

## Purpose

1. For covered episodes, the time from the annotated onset window to the first cue-on window (0 s when the cue is already on in the window before onset).
2. Composition of the registered metric "timely_lead1" (activation run that started 2 to 8 windows before the onset window): runs that started outside a preceding episode and its post-offset zone (new cue) versus runs that started during a preceding episode or its post-offset zone (carried over).

## Procedure

Registered cue states at seed 42 of the primary rule and of controller M, all five cohorts (`results_mx2/MX2_per_window_<dataset>_seed42.csv.gz`, `results_mx2_external/EV_per_window_<cohort>_seed42.csv.gz`). Episode definitions, segments and run starts as in `mx_new_controllers.Ev.counts`; the script recomputes coverage and timely_lead1 and reports them next to the values from `Ev.counts` as a consistency check. Reported per cohort and controller: episodes, coverage, cue on at onset (% of episodes), median, 75th and 90th percentile latency among covered episodes, timely_lead1 and its split into new and carried-over cues.

Script: `mx2_latency_lead_split.py`. Output: `results_latency_lead_split/LAT_T1_latency_lead_split.csv`.
