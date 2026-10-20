# Addendum to the external validation protocol: additional random seeds

Status: written and committed while the seed-42 analysis of `MX_EXTERNAL_VALIDATION.md` (archived in v1.5.0, DOI 10.5281/zenodo.23488586; script commit a3511f8) was running and before any of its results were available. Nothing in the protocol, the seed-42 analysis or its hypotheses is changed.

## Decision

Controller M and the comparator rule are additionally run with seeds 1, 2, 3 and 4 on both external cohorts (tdcsfog and defog), exactly as for seed 42 (the seed controls the random forests and the mutual-information subsample). The script `mx2_external_seeds.py` is committed before it is executed and reuses the feature caches and functions of `mx2_external.py` without modification.

## Rationale

MX-1 and MX-2 showed that false alarms per hour of the same controller can vary up to about 1.7-fold across random-forest seeds. EV1 is evaluated on seed 42; the additional seeds quantify how stable the external-validation estimates are.

## Status of the additional seeds

Robustness analysis only. EV1 to EV4 remain evaluated on seed 42 as pre-specified. For each seed, the following are reported: per-cohort outcomes of M and of the comparator, the per-cohort and pooled paired differences with patient-level bootstrap CIs, and the range across all five seeds of false alarms per hour, coverage, cue-on time and timely activation with lead of at least 1 s. All seeds are reported.
