# FoG state-transition prediction and hysteresis cue control

Code and result files for the manuscript *"Subject-Independent Evaluation of Freezing of Gait State Transitions and Hysteresis-Based Cue Control in Parkinson's Disease: A Two-Dataset Study"* (submitted to Biomedical Engineering Letters).

**This release contains exactly the code that produced the reported results.** All band-pass
filtering is causal (forward-only Butterworth); no non-causal (zero-phase) filtering routine
is included. Run `python verify_release.py` to check this in seconds.

## 1. Quick verification (no data needed)
```
pip install -r requirements.txt
python verify_release.py
```
It checks that the filter is causal, prints its group delay (20-68 ms in the 2-15 Hz band),
confirms that no script uses non-causal filtering, and checks the shipped results against the
manuscript.

## 2. Data
| Dataset | Source |
|---|---|
| DAPHNET Freezing of Gait | UCI, https://doi.org/10.24432/C56K78 |
| Multimodal FoG dataset (Li 2021), filtered data | Mendeley Data, https://doi.org/10.17632/r8gmbtv7w2.3 |

Set the data locations once in `paths.py`, or through the environment variables
`FOG_DAPHNET`, `FOG_LI_RAW` and `FOG_LI_CONVERTED`.

**Pre-processing limitations (Li 2021, as stated in the manuscript):** the dataset authors supply
the accelerometer signals low-pass filtered and normalised (implementation not specified), and our
polyphase resampling from 500 Hz to 64 Hz (`li_convert.py`) uses a symmetric anti-aliasing filter
with about 0.16 s of look-ahead. All subsequent processing is causal.

## 3. Full reproduction (run in this order)
| Step | Script | Output folder | Time* |
|---|---|---|---|
| 1 | `fog_rerun_causal.py` | `results_causal/` DAPHNET: protocol decomposition P0-P3, decoders, event metrics | 1.5 h |
| 2 | `fog_fix2.py` | P0 on real windows, surrogate test, threshold sweep | 0.5 h |
| 3 | `fog_fix3.py` | classifiers, label lengths, event-level tuning | 2.5 h |
| 4 | `fog_improve.py` | `results_improve/` ablation C0-C3 | 2-4 h |
| 5 | `fog_final_c1.py` | temporal-context controller: sensors, surrogate, statistics | 1.5 h |
| 6 | `li_check.py`, `li_convert.py` | Li 2021 diagnostic and conversion | minutes |
| 7 | `run_li.py` (`resume_li.py` if interrupted) | `results_li_causal/`, `results_li_improve/` | 3-4 h |
| 8 | `fog_constrained.py` | `results_constrained/` controller with specificity floor, both datasets | 2-3 h |
| 9 | `fog_sensors_final.py` | sensor comparison with the final controller | 1-1.5 h |
| 10 | `final_tests.py` | per-patient tests vs RF and HMM | seconds |

\*Standard laptop. Shared modules: `fog_pipeline.py` (loading, causal filtering, features, labelling),
`fog_study.py` (decoders, hysteresis, event metrics), `fog_stats.py` (statistics), `li_io.py` (reader).

## 4. Result files (`results/`) and where they appear in the manuscript
| Manuscript item | File |
|---|---|
| Table 3, Fig. 3 | `results_causal/E1_protocols.csv`, `F2_A_P0_real_only.csv`, `CM_*.csv` (and `results_li_causal/`) |
| Table 4 | `E2_decoders.csv` |
| Table 5 | `E3_event_level.csv`; proposed controller: `results_constrained/B1_summary.csv` |
| Fig. 5 | `results_constrained/B2_surrogate_*_C0spec.csv` |
| Table 6 | `results_constrained/B5_final_tests.csv` |
| Table 7 | `results_constrained/S_*.csv` |
| Online Resource | `I1_ablation_shank.csv`, `F3_*.csv`, `B4_tests_*.csv`, `B3_per_subject_*.csv` (per patient), `B_choices_*_C0spec.csv` (thresholds per fold) |

## 5. Reproducibility notes
* Feature selection, oversampling and threshold tuning are fitted on training subjects inside each
  leave-one-subject-out fold (steps P0-P1 deliberately reconstruct the original protocol).
* Fixed random seed (`Config.seed = 42`). Exact package versions: `environment_used.txt`.

## 6. Citation
See `CITATION.cff`. Archived release DOI: to be added (Zenodo).
