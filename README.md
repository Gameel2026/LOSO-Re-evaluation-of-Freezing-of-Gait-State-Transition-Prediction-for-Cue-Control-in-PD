# Subject-independent evaluation of freezing-of-gait cue control

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23080695.svg)](https://doi.org/10.5281/zenodo.23080695)

Code, analysis plans and result files for a method article submitted to *MethodsX*. The method has two parts:

1. a subject-independent, leakage-free evaluation framework for freezing-of-gait (FoG) transition labelling and cue control, built as a stepwise protocol decomposition (P0 to P3), with event-level burden measures (episode coverage, timely activation, false alarms per hour, cue-on time), a within-segment circular-shift chance reference and patient-level bootstrap confidence intervals;
2. a hysteresis cue controller evaluated under this framework, with pre-specified, time-stamped analysis plans and a pre-specified external validation on two independent cohorts.

All band-pass filtering is causal (forward-only Butterworth). `python verify_release.py` checks this in seconds without data.

## 1. Reference environment

Results depend on the library versions (random-forest implementations differ between scikit-learn releases). All reported results were produced with:

| Package | Version |
|---|---|
| Python | 3.8.5 |
| numpy | 1.19.2 |
| pandas | 1.1.3 |
| scikit-learn | 1.3.2 |
| scipy | 1.5.2 |
| imbalanced-learn | 0.12.4 |
| joblib | 1.4.2 |

```
pip install -r requirements.txt
```
The deep-learning scripts (`fog_deep.py`, `fog_lstm_controller.py`) additionally need PyTorch. The full package list of the original environment is in `environment_used.txt`.

## 2. Data

| Dataset | Role | Source |
|---|---|---|
| DAPHNET Freezing of Gait | Development | UCI, https://doi.org/10.24432/C56K78 |
| Multimodal FoG dataset (filtered data) | Development | Mendeley Data, https://doi.org/10.17632/r8gmbtv7w2.3 |
| FoG-STAR | Development (pre-specified evaluation) | Zenodo, https://doi.org/10.5281/zenodo.17838806 (`sensor_data.csv`) |
| tdcsfog (laboratory) and defog (home), lower-back sensor | External validation | Zenodo, https://doi.org/10.5281/zenodo.12565292 (`Competition Dataset.zip`, labelled training files; CC BY 4.0); described in Salomon et al., Nat Commun 2024, https://doi.org/10.1038/s41467-024-49027-0 |

Set the DAPHNET and Multimodal FoG locations in `paths.py` (or the environment variables `FOG_DAPHNET`, `FOG_LI_RAW`, `FOG_LI_CONVERTED`). The FoG-STAR paths are set at the top of `fogstar_convert.py` and `run_fogstar.py`. The external cohorts are converted with `kaggle_convert.py`: set `ROOT` and `OUT` to local folders and `DRY_RUN = False`.

External-cohort conversion rules (`kaggle_convert.py`): labelled training files only; FoG = StartHesitation, Turn or Walking (Event for the notype recordings of the defog study); defog samples with Valid or Task equal to False excluded; units converted to mg; lower-back axes AccAP, AccV, AccML; 128 Hz (tdcsfog) and 100 Hz (defog) resampled to 64 Hz by polyphase filtering; FoG segments separated by less than 2 s within a valid stretch merged into one episode; files shorter than 2 s skipped.

## 3. Analysis plans and identifiers

Each plan was committed before the analysis it describes was executed; results were committed afterwards. Post hoc analyses are labelled as such in their plan files.

| Stage | File(s) | Commit | Release / DOI |
|---|---|---|---|
| Pre-specified FoG-STAR evaluation | `PREREGISTRATION_FoGSTAR.md` | 5a4b71d | |
| Analysis plan MX-1 | `MX_AMENDMENT_1.md` | 20fcbbe | v1.1.0, 10.5281/zenodo.23481294 |
| MX-1 script | `mx_new_controllers.py` | 2574849 | |
| MX-1 results | `Results/results_mx1/` | | v1.2.0, 10.5281/zenodo.23484310 |
| Analysis plan MX-2 | `MX_AMENDMENT_2.md` | 1f31324 | v1.3.0, 10.5281/zenodo.23484328 |
| MX-2 script | `mx2_controller.py` | 5a3a96f | |
| MX-2 results | `Results/results_mx2/` | | v1.4.0, 10.5281/zenodo.23488425 |
| External validation plan | `MX_EXTERNAL_VALIDATION.md` | 4eb4d30 | v1.5.0, 10.5281/zenodo.23488586 |
| External validation script | `mx2_external.py` | a3511f8 | |
| External-cohort conversion script | `kaggle_convert.py` | 7680d71 | |
| Additional seeds (robustness) | `MX_EXTERNAL_VALIDATION_ADDENDUM_SEEDS.md`, `mx2_external_seeds.py` | a7557de, 2c94434 | |
| Post hoc: threshold grid sensitivity | `MX_EXTERNAL_GRID_SENSITIVITY.md`, `mx2_external_grid_sensitivity.py` | d9469c8 (results f356ec3) | |
| Post hoc: pre-onset separability | `MX_EXTERNAL_SEPARABILITY.md`, `mx2_external_separability.py` | 575675f (results a9c16f7) | |
| Post hoc: operating-point chance reference | `MX_EXTERNAL_OPERATING_SURROGATE.md`, `mx2_external_operating_surrogate.py` | f7c6be9 (results 1ca0ea6) | |
| Post hoc: decomposition, reference strategies, per patient | `MX_EXTERNAL_DECOMPOSITION.md`, `mx2_external_decomposition.py` | e398f1c (results 8dca24e) | |
| Post hoc: segment structure | `MX_SEGMENT_STRUCTURE.md`, `mx2_segment_structure.py` | 7fccef1 (results 2e584ae) | |
| Post hoc: within-segment separability and onset-locked profile | `MX_ONSET_PROFILE.md`, `mx2_onset_profile.py` | 694bbb0 (results 8792a08) | |
| Post hoc: activation latency and early-activation composition | `MX_LATENCY_LEAD_SPLIT.md`, `mx2_latency_lead_split.py` | 7151500 (results c755a9f) | |
| Post hoc: coverage chance reference, episode durations, false-alarm time base | `MX_FINAL_DESCRIPTIVES.md`, `mx2_final_descriptives.py` | bbb2322 (results 66cf50b) | |
| Post hoc: protocols P0 to P3 across seeds with patient-bootstrap intervals | `MX_LADDER_UNCERTAINTY.md`, `mx_ladder_uncertainty.py` | d3bac56 (script configuration corrected before the final run, a3e8947) | |
| External validation results (seed 42 and seeds 1 to 4) | `Results/results_mx2_external/` | 10c0291 | v1.6.0 |
| Post hoc results | `Results/<folder named in each plan>` | | v1.6.0 |

## 4. Reproducing the method article

Run from the repository folder (all scripts use paths relative to the working folder) in this order:

| Step | Script | Output folder |
|---|---|---|
| 1 | development pipeline, section 5 (provides `results_causal/`, `results_li_causal/`, `results_fogstar_causal/` feature caches) | see section 5 |
| 2 | `mx_reference_rerun.py` | `results_methodsx/` (per-window outputs of the primary controller, reference for steps 4 to 6) |
| 3 | `mx_reference_analyses.py` | `results_methodsx/analyses/` (reference strategies, episode partition, tolerance zone, minimum lead, patient-level chance reference, matched false-alarm comparison, per patient) |
| 4 | `mx_new_controllers.py` | `results_mx1/` |
| 5 | `mx2_controller.py` | `results_mx2/` |
| 6 | `kaggle_convert.py`, then `mx2_external.py` | `results_mx2_external/` (seed 42) |
| 7 | `mx2_external_seeds.py` | `results_mx2_external/` (seeds 1 to 4 and summaries `EVS_T1`, `EVS_T3`, `EVS_T6`) |
| 8 | post hoc scripts of section 3 | one folder per script, as named in each plan |

Steps 4 to 8 save checkpoints and resume after an interruption. On a standard laptop, one external-validation seed takes about 7 hours for defog (45 folds) and a comparable time for tdcsfog (62 folds); `mx_ladder_uncertainty.py` takes about two hours for five seeds and three development cohorts.

External-validation outputs (`Results/results_mx2_external/`): `EV_T2` to `EV_T5` (seed 42: controllers with CIs, paired differences, hypotheses EV1 to EV4, chance reference), `EV_choices_<cohort>.csv` (selected operating point per fold), `EV_counts_<cohort>.pkl` (per-patient event counts), `EV_per_window_<cohort>_seed42.csv.gz` (held-out cue states per window); `EVS_choices_<cohort>_seed<k>.csv` and `EVS_counts_<cohort>_seed<k>.pkl` for seeds 1 to 4; `EVS_T1` (all seeds), `EVS_T3` (paired differences, all seeds) and `EVS_T6` (range across seeds). Hypotheses EV1 to EV4 are evaluated on seed 42 (`EV_T4`); `EVS_T3` redraws the bootstrap in one sequence across seeds, so its seed-42 intervals differ slightly from `EV_T3`, while point estimates are identical. The seed controls the random forests and the mutual-information subsample.

## 4a. Applying the method to new data

1. Convert each recording to the DAPHNET text layout (time, three accelerometer axes per sensor, label 0 = not annotated, 1 = no FoG, 2 = FoG), as `kaggle_convert.py` does for the external cohorts; resample to 64 Hz.
2. Build windows, features and labels with `fog_pipeline.py` (0.5 s windows within contiguous annotated segments; a window is FoG if at least 0.3 s is FoG; Pre-FoG 4 s and Post-FoG 3 s), cached as in `fog_rerun_causal.data_for`.
3. Evaluate with leave-one-subject-out folds, fitting feature selection, oversampling and threshold selection on the training patients only (`mx2_external.py` shows the complete loop for a new cohort; add the cohort to `COHORTS`).
4. Report the burden pair (false alarms per hour and cue-on time) with coverage and activation at least two windows before the onset window, each against the within-segment circular-shift reference (`mx_new_controllers.Ev`), with patient-level bootstrap intervals and the range across seeds.
5. Customisation points: window length, Pre-FoG/Post-FoG intervals (which also define the false-alarm tolerance zone), threshold grid, specificity floor, false-alarm budget, ensemble size and hold set (`mx2_controller.py`).

## 5. Development pipeline

| Script | Purpose | Output folder |
|---|---|---|
| `fog_rerun_causal.py` | DAPHNET: protocols P0 to P3 for the four-class random forest, decoders, event metrics | `results_causal/` |
| `fog_fix2.py` | chance reference, threshold sweep, P0 scored on real windows | `results_causal/` |
| `fog_fix3.py` | alternative classifiers, label lengths, event-level tuning | `results_causal/` |
| `fog_improve.py` | controller ablation | `results_improve/` |
| `fog_final_c1.py` | temporal-context controller | |
| `li_check.py`, `li_convert.py`, `run_li.py`, `resume_li.py` | Multimodal FoG conversion and analyses | `results_li_causal/`, `results_li_improve/` |
| `fog_constrained.py` | controller with specificity floor | `results_constrained/` |
| `fog_sensors_final.py` | sensor locations | |
| `final_tests.py` | per-patient tests | |
| `fog_floor_sensitivity.py` | specificity-floor sensitivity | `results_floor/` |
| `fog_deep.py`, `fog_lstm_controller.py` | deep-learning baselines and LSTM-driven controller (PyTorch) | `results_deep/`, `results_lstm_controller/` |
| `fogstar_convert.py`, `run_fogstar.py` | FoG-STAR conversion and pre-specified evaluation | `results_fogstar_causal/`, `results_fogstar_controller/` |
| `fogstar_check.py`, `fogstar_activity.py`, `fogstar_stratified.py`, `fog_precursor.py` | FoG-STAR descriptive and post hoc analyses; pre-onset separability | `results_precursor/` |
| `fog_auprc.py`, `fog_feature_sensitivity.py`, `fog_nested_features.py`, `fog_release_sensitivity.py` | further sensitivity analyses | `results_auprc/`, `results_features/`, `results_nested/`, `results_release_sensitivity/` |

Shared modules: `fog_pipeline.py` (loading, causal filtering, windowing, features, labelling), `fog_study.py` (decoders, hysteresis, event metrics), `fog_stats.py` (statistics), `li_io.py` (reader), `paths.py`.

## 6. Notes

* Feature selection, oversampling, standardisation and threshold selection are fitted on training patients inside each leave-one-subject-out fold (protocols P0 and P1 use record-wise folds by design).
* Event metrics, hysteresis states and the chance reference operate within contiguous annotated segments.
* Result files of all steps are collected under `Results/`, in folders with the names written by the scripts.
* Checkpoint folders (`ckpt_*`, `checkpoints/`) are only used to resume interrupted runs.

## 7. Licence and citation

Code: MIT licence (`LICENSE`). Datasets remain under the licences of their sources. See `CITATION.cff`. Concept DOI (all versions): https://doi.org/10.5281/zenodo.23080695
