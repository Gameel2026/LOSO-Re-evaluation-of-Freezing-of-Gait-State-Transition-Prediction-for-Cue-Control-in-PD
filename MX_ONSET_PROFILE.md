# Post hoc diagnostic analysis: within-segment separability and onset-locked score profile

Status: post hoc, diagnostic. Written after the seed-42 external results and the exploratory analyses (commits d9469c8, 575675f, f7c6be9, e398f1c, 7fccef1) were available, and committed together with its script before the script was executed. EV1 to EV4 and their verdicts are not changed. Reported in the Supplementary material.

## Motivation

In the home cohort (defog), activation timing of controller M did not exceed the circular-shift reference at any of seven operating points or within long segments, although the pooled pre-onset AUC of the score was 0.65. The circular-shift reference is applied within recording segments, so timing above the reference requires the score to rise before onset relative to the rest of the same segment. A pooled AUC can also reflect differences between segments or patients. The analysis measures (1) separability within segments and (2) the time course of the score around onset relative to the same segment.

## Expectation stated before execution

If the home result reflects the absence of a within-segment rise of the score at least 1 s before onset, the within-segment AUC (lead of at least 1 s) in defog is close to 0.5 with a CI including 0.5, and the onset-locked profile in defog does not rise above the segment baseline before the last window before onset, whereas tdcsfog shows a within-segment AUC above 0.5. All results are reported whatever their direction.

## Procedure

Held-out per-window scores of controller M at seed 42 in all five cohorts (`results_mx2/MX2_per_window_<dataset>_seed42.csv.gz`, `results_mx2_external/EV_per_window_<cohort>_seed42.csv.gz`). Window sets as in `MX_EXTERNAL_SEPARABILITY.md` (pre-onset windows 2 to 8 before onset, within the segment, not FoG, not in the post-offset zone of a preceding episode; No-FoG windows outside every episode zone).

1. Within-segment AUC: pre-onset windows are compared only with No-FoG windows of the same segment; pair counts are summed over segments (equivalent to a segment-stratified Mann-Whitney AUC). Patient-level bootstrap CI (5,000 resamples).
2. Onset-locked profile: for each episode, the score at offsets −4 s to +2 s (−8 to +4 windows) minus the median score of the No-FoG windows of the same segment; pre-onset offsets exclude FoG windows and windows before the segment start or within the post-offset zone of a preceding episode. Mean per offset with patient-level bootstrap CI.

Script: `mx2_onset_profile.py`. Outputs: `results_onset_profile/PROF_T1_within_segment_auc.csv`, `PROF_T2_onset_locked_profile.csv`.
