# FoG state-transition prediction and hysteresis cue control

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23080695.svg)](https://doi.org/10.5281/zenodo.23080695)

Code and result files for the manuscript *"Subject-Independent Evaluation of Freezing-of-Gait Transition Prediction in Parkinson’s Disease: From Window-Level to Event-Level Assessment for Wearable Cueing"* (submitted to *Health Information Science and Systems*).

**This release contains exactly the code that produced the reported results.** All band-pass
filtering is causal (forward-only Butterworth); no non-causal (zero-phase) filtering routine
is included. Run `python verify_release.py` to check this in seconds.

## 1. Quick verification (no data needed)
```
pip install -r requirements.txt
python verify_release.py
# the deep-learning scripts additionally need: pip install torch
```
It checks that the filter is causal, prints its group delay (20-68 ms in the 2-15 Hz band),
confirms that no script uses non-causal filtering, and checks the shipped results against the
manuscript.

## 2. Data
| Dataset | Role in the study | Source |
|---|---|---|
| DAPHNET Freezing of Gait | Primary (development) | UCI, https://doi.org/10.24432/C56K78 |
| Multimodal FoG dataset, filtered data | Replication | Mendeley Data, https://doi.org/10.17632/r8gmbtv7w2.3 |
| FoG-STAR | Pre-registered independent evaluation | Zenodo, https://doi.org/10.5281/zenodo.17838806 (`sensor_data.csv`) |

Set the DAPHNET and Multimodal FoG locations once in `paths.py`, or through the environment
variables `FOG_DAPHNET`, `FOG_LI_RAW` and `FOG_LI_CONVERTED`. The FoG-STAR paths are set at the
top of `fogstar_convert.py` and `run_fogstar.py`.

**Pre-processing limitations (Multimodal FoG, as stated in the manuscript):** the dataset authors
supply the accelerometer signals low-pass filtered and normalised (implementation not specified),
and the polyphase resampling from 500 Hz to 64 Hz (`li_convert.py`) uses a symmetric anti-aliasing
filter with about 0.16 s of look-ahead. FoG-STAR is resampled from 60 Hz to 64 Hz in the same way.
All subsequent processing is causal.

## 3. Pre-registration of the FoG-STAR evaluation
`PREREGISTRATION_FoGSTAR.md` fixes the conversion rules, the analysis (identical to the manuscript,
specificity floor 0.70) and hypotheses H1-H4 for FoG-STAR. It was committed to this repository
**before** any model was trained or evaluated on FoG-STAR; its commit date documents this. Later
commits add code and results but do not modify the plan.

## 4. Full reproduction (run in this order)
| Step | Script | Output folder | Time* |
|---|---|---|---|
| 1 | `fog_rerun_causal.py` | `results_causal/` DAPHNET: four-class RF under LOSO, decoders, event metrics (also the additional window-level comparison, see section 6) | 1.5 h |
| 2 | `fog_fix2.py` | surrogate test, threshold sweep (also the real-window check of the additional comparison, see section 6) | 0.5 h |
| 3 | `fog_fix3.py` | alternative classifiers, label lengths, event-level tuning | 2.5 h |
| 4 | `fog_improve.py` | `results_improve/` controller ablation C0-C3 | 2-4 h |
| 5 | `fog_final_c1.py` | temporal-context controller: sensors, surrogate, statistics | 1.5 h |
| 6 | `li_check.py`, `li_convert.py` | Multimodal FoG diagnostic and conversion | minutes |
| 7 | `run_li.py` (`resume_li.py` if interrupted) | `results_li_causal/`, `results_li_improve/` | 3-4 h |
| 8 | `fog_constrained.py` | `results_constrained/` controller with specificity floor, DAPHNET and Multimodal FoG | 2-3 h |
| 9 | `fog_sensors_final.py` | sensor comparison with the final controller | 1-1.5 h |
| 10 | `final_tests.py` | per-patient tests vs RF and HMM | seconds |
| 11 | `fog_floor_sensitivity.py` | `results_floor/` controller with specificity floors 0.60-0.80 and without a floor | 1-2 h |
| 12 | `fog_deep.py` | `results_deep/` 1D-CNN and LSTM baselines (0.5 s input; 2 s and 4 s context); needs PyTorch | 1-2 h |
| 13 | `fog_lstm_controller.py` | `results_lstm_controller/` controller driven by the 4 s LSTM (secondary analysis); needs PyTorch | 1-2 h |
| 14 | `fogstar_convert.py`, `run_fogstar.py` | `results_fogstar_causal/`, `results_fogstar_controller/` pre-registered evaluation on FoG-STAR | < 1 h |
| 15 | `fogstar_check.py`, `fogstar_activity.py` | post-hoc descriptive analysis of pre-onset recording and activity (no model outputs) | minutes |
| 16 | `fogstar_stratified.py`, `fog_precursor.py` | further post-hoc analyses of the FoG-STAR result (stratification by pre-onset activity; pre-freezing separability) | < 30 min |
| 17 | `fog_auprc.py` | `results_auprc/` AUPRC on the three datasets (reserve analysis, not reported in the manuscript) | 0.5-1 h |
| 18 | `fog_feature_sensitivity.py` | `results_features/` number of selected features (10, 23, 40, 72) for the four-class RF (LOSO) and the final controller, development datasets; resumes after an interruption | 2-3 h |
| 19 | `fog_nested_features.py` | `results_nested/` number of features chosen within each training fold, on all three datasets, compared with the 23-feature controller in the same folds; resumes after an interruption | 2-3 h |

\*Standard laptop. Shared modules: `fog_pipeline.py` (loading, causal filtering, features, labelling),
`fog_study.py` (decoders, hysteresis, event metrics), `fog_stats.py` (statistics), `li_io.py` (reader).

## 5. Result files and where they appear in the manuscript
| Manuscript item | Content | File |
|---|---|---|
| Table 5; Figs. 4-5 | Four-class RF under LOSO; confusion matrices | `results_causal/E1_protocols.csv` (row `P3 + in-fold feature selection`), `CM_P3.csv`, `CM_hmm.csv` (same files in `results_li_causal/`) |
| Table 6 | Window-level decoders | `results_causal/E2_decoders.csv`, `results_li_causal/E2_decoders.csv` |
| Table 7; Fig. 6 | Event-level cue performance | `E3_event_level.csv`; proposed controller: `results_constrained/B1_summary.csv` |
| Table 8 | Per-patient results of the proposed controller | `results_constrained/B3_per_subject_*.csv` |
| Fig. 7 | Surrogate test | `results_constrained/B2_surrogate_*_C0spec.csv` |
| Table 9 | Effect of the specificity floor | `results_constrained/B1_summary.csv`, `B4_tests_*.csv` |
| Table 10 | Thresholds per fold | `results_constrained/B_choices_*_C0spec.csv` |
| Table 11 | Specificity-floor sensitivity | `results_floor/FS1_summary.csv`, `FS2_surrogate.csv`, `FS4_tests.csv` |
| Table 12 | Per-patient comparison with RF and HMM | `results_constrained/B5_final_tests.csv` |
| Table 13 | Sensor location | `results_constrained/S_sensors_summary.csv`, `S_surrogate_*.csv`, `S_tests_sensors.csv` |
| Table 14 | Alternative classifiers and label lengths | `results_causal/F3_*.csv` |
| Table 15 | Controller ablation | `results_improve/I1_ablation_shank.csv`, `results_li_improve/` |
| Fig. 8 | Threshold sweep | `results_causal/F2_B3_threshold_sweep.csv`, `F2_threshold_sweep.png` |
| Table 16 | Number of selected features (fixed and nested) | `results_features/FK1_fourclass.csv`, `FK2_controller.csv`, `FK3_surrogate.csv`, `FK4_tests.csv`; `results_nested/N1_summary.csv`, `N2_surrogate.csv`, `N4_tests.csv`, `N5_choices.csv` |
| Table 17; Fig. 9 | Deep-learning baselines | `results_deep/DL1_summary.csv`, `DL3_tests.csv`, `DL4_confusion_*.csv` |
| Table 18 | LSTM-driven controller | `results_lstm_controller/LC1_summary.csv`, `LC2_surrogate.csv`, `LC4_tests.csv` |
| Tables 19-20 | Pre-registered evaluation on FoG-STAR (H1-H3 in the manuscript; H4 in section 6) | `results_fogstar_causal/E1_protocols.csv` (row `P3 + in-fold feature selection`), `E2_decoders.csv`, `E3_event_level.csv`; `results_fogstar_controller/X1_summary.csv`, `X2_surrogate.csv`, `X3_per_subject.csv`, `X4_tests.csv` |
| Section 3.9 (post hoc) | Pre-onset recording, activity, stratified and separability analyses | `fogstar_check.csv`, `fogstar_activity.csv`, `results_fogstar_controller/X5_stratified.csv`, `results_precursor/P1_separability.csv` |
| Fig. 10 | Summary across the three datasets | values from the files above |

## 6. Additional analyses not reported in the manuscript
* **Window-level comparison (steps P0-P2).** `fog_rerun_causal.py` and `run_fogstar.py` also evaluate the
  four-class RF under window-level 10-fold cross-validation with SMOTE applied before partitioning (P0),
  with SMOTE inside the training folds (P1) and with a subject-wise split (P2); `fog_fix2.py` scores P0 on
  real test windows only. These rows are in `E1_protocols.csv` and `F2_A_P0_real_only.csv`. They are not
  part of the manuscript, which reports subject-independent (LOSO) evaluation only.
* **Pre-registered hypothesis H4.** H4 compared transition recall under LOSO with P0 on FoG-STAR. Its
  result is in `results_fogstar_causal/E1_protocols.csv` (rows `P0` and `P3`) and is printed by `run_fogstar.py`.
* **AUPRC (step 17)**: `results_auprc/`.

## 7. Reproducibility notes
* Feature selection, oversampling, standardisation and threshold tuning are fitted on training
  patients inside each leave-one-subject-out fold (only the additional window-level steps P0-P1 of
  section 6 use window-level folds, by design).
* Each dataset is evaluated separately with leave-one-subject-out validation; no model is trained
  on one dataset and tested on another.
* The folders `results_features/checkpoints/` and `results_nested/checkpoints/` hold temporary files used only to resume an interrupted run and are not part of the results.
* Fixed random seed (`Config.seed = 42`). Exact package versions: `environment_used.txt`.

## 8. Citation
See `CITATION.cff`. Archived release: https://doi.org/10.5281/zenodo.23080695
