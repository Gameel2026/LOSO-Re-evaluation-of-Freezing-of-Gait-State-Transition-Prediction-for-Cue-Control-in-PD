# Pre-specified amendment MX-2: burden-matched, dominance-seeking threshold selection with a variance-reduced score

Status: written after the results of amendment MX-1 (commit 20fcbbe, DOI 10.5281/zenodo.23481294; analysis script commit 2574849) were known, and committed before the controller described in Section 3 was run on any dataset. The commit time stamp of this file is the reference; the analysis script is committed separately before execution. All results of MX-2 are reported, irrespective of outcome, at least in the supplementary material.

## 1. Information available when this amendment was written

In addition to the information declared in MX-1:

1. All MX-1 outputs (tables T1 to T7, per-window outputs for seed 42), produced in the reference software environment with the reproduction check passed for all three datasets.
2. Operating curves of the union score with and without the 3 s hold, evaluated on the test patients (MX-1 table T6 and an equivalent sweep without hold), including curves constrained to the cue-on time of the frozen primary controller.

Because these curves were computed on the same patients that MX-2 is evaluated on, MX-2 is a pre-specified analysis informed by MX-1, not an independent confirmation. This is stated wherever MX-2 results are reported.

## 2. Rationale (from MX-1)

- The event-level rule with a fixed 3 s hold reduced false alarms per hour in all three datasets (differences -89, -45 and -63 per hour, all 95% CIs below zero), but the fixed budget of 60 per hour placed the operating point below the coverage of the primary controller; H1 of MX-1 failed on its coverage margin.
- Coverage CIs for paired differences had half-widths of about 6 to 10 percentage points (8 to 16 patients with episodes per dataset). A 5-point non-inferiority margin is therefore not attainable per dataset at this sample size even for a controller identical to the comparator; margins below are set from this precision, and the key superiority test pools the three cohorts.
- The false-alarm rate of the primary controller varied up to 1.7-fold across five random-forest seeds (DAPHNET 75 to 127 per hour), indicating that 100 trees leave substantial threshold-level variance.
- The hold improved the false-alarm/coverage trade-off at matched cue-on time in DAPHNET and FoG-STAR but not in Multimodal FoG (high FoG prevalence), so a fixed hold is not appropriate for all cohorts.
- Training targets other than the primary union target (D) and the anticipation model (A) did not improve outcomes and are not carried forward.

## 3. Controller M (burden-matched, dominance-seeking)

Common elements are unchanged from the primary analysis and MX-1: leave-one-subject-out; 23 features ranked by mutual information on training patients; union training target (Pre-FoG, FoG and Post-FoG versus No-FoG); balanced class weights; inner out-of-fold scores from a three-fold subject-grouped split of the training patients.

**Variance-reduced score.** Random forest with 500 trees in the final model and 250 trees in the inner folds (primary: 100 and 50).

**Candidate decision rules.** Hysteresis (theta_on 0.30 to 0.90 step 0.05; theta_off 0.10 to theta_on step 0.10) combined with a post-activation hold of 0, 1, 2 or 3 s (0, 2, 4 or 6 windows; hold semantics as in MX-1).

**Reference operating point (per fold).** The primary selection rule (window F1 with training specificity of at least 0.70, no hold) is applied to the same inner out-of-fold scores; its inner coverage C_ref and inner cue-on time T_ref define the reference.

**Selection (per fold, on pooled inner out-of-fold scores of the training patients).**
1. Feasible set: candidates with inner coverage at least C_ref, inner cue-on time at most T_ref, and specificity outside the tolerance zone of at least 0.70 (the candidate dominates or equals the reference on coverage and cue burden).
2. Within the feasible set, choose the candidate with the fewest false alarms per hour; ties are broken by higher timely activation with lead of at least 1 s, then by shorter hold.
3. If the feasible set is empty, choose the candidate with the highest inner coverage among those with cue-on time at most T_ref; if none, apply the primary rule.

No other parameter is learned or tuned. Seeds: 42 (primary result) and 1 to 4 (robustness), each controlling the random forests and the mutual-information subsample.

## 4. Outcomes and comparator

Comparator: the frozen primary controller (seed 42). Outcomes and definitions as in MX-1 (coverage, frozen timely activation, timely activation with lead of at least 1 s, cue active at onset, false alarms per hour, cue-on time). Uncertainty: 5,000-sample patient-level bootstrap of paired differences, resampling patients within each dataset; the pooled three-cohort estimate resamples within each dataset and combines episode and hour totals across datasets. Chance reference: 200 circular shifts within recording stretches.

## 5. Hypotheses and decision rules (fixed before running)

- H1 (false alarms, pooled): in the pooled three-cohort estimate, M has fewer false alarms per hour than the primary controller (95% CI of the difference below zero). Per-dataset differences and CIs are reported descriptively.
- H2 (coverage, non-inferiority): in each dataset, the lower 95% CI bound of the coverage difference (M minus primary) is above -10 percentage points.
- H3 (cue burden, non-inferiority): in each dataset, the upper 95% CI bound of the cue-on time difference is below +5 percentage points.
- H4 (anticipation, non-inferiority): in each dataset, the lower 95% CI bound of the difference in timely activation with lead of at least 1 s is above -10 percentage points.
- H5 (stability): in each dataset, the range of false alarms per hour across the five seeds is narrower for M than for the primary controller (primary ranges taken from MX-1 table T7, recomputed in the same run).

Each hypothesis is reported as supported, not supported or, for non-inferiority tests whose CI is wider than the margin, inconclusive. No multiplicity adjustment is applied; results are interpreted per hypothesis. All results are reported whether or not the hypotheses are supported.
