# Pre-specified external validation of controller M (amendment MX-2) on two independent cohorts

Status: written after the MX-2 results were known and committed before controller M or the comparator rule was run on the cohorts below. The commit time stamp of this file is the reference; the analysis script (`mx2_external.py`) is committed separately before execution. All results are reported irrespective of outcome.

## 1. Purpose

MX-2 showed that controller M preserved coverage, cue-on time and anticipatory activation relative to the primary rule in DAPHNET, Multimodal FoG and FoG-STAR, with a pooled reduction in false alarms of 21.9 per hour whose 95% CI included zero (-73.0 to +21.7; 44 patients). This protocol tests the same, unchanged method on independent cohorts that were not used to design it.

## 2. Cohorts and data preparation

- tdcsfog: laboratory FoG-provoking walking protocol, 62 patients (41 with FoG), 15.3 h, 949 episodes.
- defog: home recordings, restricted to samples annotated as valid and recorded during the FoG-provoking task protocol (Valid and Task both true), 45 patients (43 with FoG), 20.9 h, 689 episodes; includes the defog files with untyped FoG events ("notype").

Source: the labelled training files of the Kaggle/Zenodo FoG competition data (CC BY 4.0). Conversion to the DAPHNET text format uses `kaggle_convert.py` unchanged (lower-back accelerometer to the trunk columns, units in mg, resampling to 64 Hz, FoG as the union of the annotated FoG types, merging of FoG segments separated by less than 2 s within a valid stretch). The figures above are taken from the conversion summary (`kaggle_runs_summary.csv`), which contains no model output. These data were converted and analysed, with a different model and a different question, in a separate study by the same authors; neither controller M nor the comparator rule of this protocol has been applied to them.

Sensor: lower-back accelerometer (the only sensor in these cohorts), whereas the three development cohorts used the shank. Features: the same 72 features computed by `fog_pipeline.py` from the trunk columns (causal filter, 0.5 s windows), cached by `fog_rerun_causal.data_for`.

## 3. Method (unchanged from MX-2)

Controller M exactly as specified in MX-2 Section 3 (500-tree random forest, 250 trees in inner folds; hysteresis with hold of 0 to 3 s; dominance-seeking selection relative to the primary rule on inner out-of-fold scores). Comparator: the primary rule (100-tree random forest, 50 in inner folds; window F1 with training specificity of at least 0.70; no hold), trained within the same leave-one-subject-out folds on the same cohort. Seed: 42 only; repeated seeds are not run because of computational cost (about 10 h per seed for both cohorts), and the MX-2 hypothesis on seed stability is therefore not tested here.

## 4. Outcomes and analysis

Outcomes and definitions as in MX-1 and MX-2. Patient-level bootstrap (5,000 resamples) of paired differences within each cohort; pooled estimates resample patients within each cohort and combine episode and hour totals across cohorts. Chance reference: 200 circular shifts within recording stretches.

## 5. Hypotheses and decision rules (fixed before running)

- EV1 (primary, independent confirmation): in the pooled estimate over tdcsfog and defog, M has fewer false alarms per hour than the comparator (95% CI of the difference below zero).
- EV2: in each cohort, the lower 95% CI bound of the coverage difference is above -10 percentage points.
- EV3: in each cohort, the upper 95% CI bound of the cue-on time difference is below +5 percentage points.
- EV4: in each cohort, the lower 95% CI bound of the difference in timely activation with lead of at least 1 s is above -10 percentage points.
- Secondary (not confirmatory, because three of the five cohorts informed the design): the false-alarm difference pooled over all five cohorts, using the MX-2 seed-42 per-window outputs for DAPHNET, Multimodal FoG and FoG-STAR.

Non-inferiority results whose CI half-width exceeds the margin and that are not supported are reported as inconclusive. No multiplicity adjustment is applied. All results are reported whether or not the hypotheses are supported.
