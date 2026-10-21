# Post hoc analysis: precision of the protocol comparison P0 to P3 across seeds and patients

Status: post hoc, precision of an existing analysis; no hypothesis is added. Written after all registered and post hoc results were available, and committed together with its script before the script was executed. All outputs are reported. EV1 to EV4, MX-1, MX-2 and their verdicts are not changed. Only development data (DAPHNET, Multimodal FoG, FoG-STAR) are used; the external cohorts are not accessed.

## Purpose

The comparison of protocols P0 to P3 (fog_rerun_causal.py) was reported from a single run at seed 42. Its variability across random seeds and its uncertainty across patients are reported here.

## Procedure

1. Protocols P0 to P3 are repeated exactly as in fog_rerun_causal.py (shank sensor, cached windows and features, hybrid 23 features for P0 to P2, mutual-information selection of 23 features inside each fold for P3, causal filtering) at seeds 42, 1, 2, 3 and 4. The seed sets the fold partition, the oversampling and the random forest. Every seed is reported. The seed-42 run is compared with the registered E1_protocols.csv values.
2. For P1 to P3, 95% confidence intervals are obtained by resampling patients with replacement (5,000 resamples, random generator seeded with the run seed), using per-patient confusion matrices of the held-out predictions. For P0, the interval is computed on the original windows only (the synthetic windows have no patient); because the synthetic windows used in training were generated from all patients, this interval is approximate.
3. Paired patient-bootstrap differences are reported for P0 (original windows) minus P1, P1 minus P2, P1 minus P3 and P3 minus P2.

## Interpretation stated before execution

Only the contrast P0 minus P1 (oversampling before partitioning against oversampling inside folds) is interpreted. The differences among P1, P2 and P3 are reported descriptively, without a claim about their size or order, in line with the main text.

Script: `mx_ladder_uncertainty.py`. Outputs (`results_ladder_uncertainty/`): `LAD_T0_seed_spread.csv`, `LAD_T1_by_seed.csv`, `LAD_T2_patient_bootstrap.csv`, `LAD_T3_paired_differences.csv`, `LAD_T4_seed42_reproduction.csv`. Checkpoint files (`ckpt_*.pkl`) are working files and are not uploaded.
