# Post hoc analysis: recording-segment structure and activation timing in long segments

Status: post hoc. Written after the seed-42 external results and the exploratory analyses (commits d9469c8, 575675f, f7c6be9, e398f1c) were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. Reported in the Supplementary material.

## Motivation

All event metrics, the hysteresis state and the circular-shift reference operate within contiguous annotated segments (stretches with a valid label). In the home cohort (defog), samples with Valid or Task equal to False are excluded during conversion, which divides recordings into many short segments. Short segments limit the pre-onset context available before an onset and restrict the range of the within-segment circular shift, which can reduce the ability of the reference to distinguish timely from untimely activation. The analysis describes the segment structure of all five cohorts and repeats the timing comparison of the registered controller M in the external cohorts within long segments only.

## Expectation stated before execution

The segment lengths were inspected during the review that motivated this note (defog shorter than the other cohorts); part 1 formalises that description. The long-segment timing (part 2) had not been computed when this note was written. If short segments limit the home timing evaluation, timely activation with lead of at least 1 s exceeds the circular-shift reference in defog when the evaluation is restricted to segments of at least 30 s or 60 s. All results are reported whatever their direction.

## Procedure

Inputs: held-out per-window files at seed 42 (`results_mx2/MX2_per_window_<dataset>_seed42.csv.gz`, `results_mx2_external/EV_per_window_<cohort>_seed42.csv.gz`).

1. Segment structure per cohort: recordings, segments, segments containing FoG, median and 10th/90th percentile segment length, episodes, percentage of onsets within 4 s (8 windows) of the segment start, percentage of onsets at the first window of a segment.
2. Long-segment timing (tdcsfog and defog): the registered cue states of controller M are evaluated within segments of at least 60 windows (30 s) and, separately, at least 120 windows (60 s). The circular-shift reference of the registered analysis is applied unchanged (`mx_new_controllers.surrogate`, 200 shifts, random generator seeded with 42). Reported: share of windows retained, episodes retained, observed value, surrogate mean, surrogate 95th percentile and one-sided p for timely activation with lead of at least 1 s (primary metric of this analysis), cue at onset and timely activation.

Script: `mx2_segment_structure.py`. Outputs: `results_segment_structure/SEG_T1_segment_structure.csv`, `SEG_T2_timing_long_segments.csv`.
