# Pre-registered analysis plan: external validation on FoG-STAR

**Written before any model was trained or evaluated on FoG-STAR.**
Commit this file (and the two scripts) to the GitHub repository *before* running `run_fogstar.py`,
so that the commit date documents that the plan preceded the results.

## Dataset
FoG-STAR (Borzì et al., Sci Data 13, 305, 2026; https://doi.org/10.5281/zenodo.17838806, CC BY 4.0):
22 people with PD (Off medication), 4 IMUs (both ankles, lower back, wrist), 60 Hz, 101 annotated FoG episodes.
It was not used in any development step of the study.

## Fixed conversion rules (`fogstar_convert.py`)
1. Accelerometer channels only; units g converted to mg (DAPHNET convention).
2. One recording ("run") per subject x session x task, in time order.
3. Ankle sensor = the ankle with fewer missing samples for that subject (a rule that does not use FoG labels);
   the lower-back sensor is converted for a secondary sensor analysis.
4. Resampling 60 Hz -> 64 Hz with a polyphase filter (up 16, down 15), as for Multimodal FoG.
5. Missing samples (NaN) are set to zero before resampling and every output sample within
   +/- 10 input samples (about 0.17 s) of a missing sample is excluded (label 0), which splits the recording.
6. FoG label = 2 where the dataset's FoG label is positive, otherwise 1.

## Fixed analysis (identical to the manuscript; nothing is tuned on FoG-STAR)
- Causal band-pass filter, 0.5 s windows, 72 features, MI top-23, random forest (100 trees), seed 42.
- Four-class labels: Pre-FoG 4 s, Post-FoG 3 s.
- Experiment 1: protocol decomposition P0-P3 and P0 on real windows.
- Experiments 2-3: decoders (per-window RF, moving average, HMM filtering, Viterbi) and event metrics.
- Experiment 4 (primary): binary RF controller with hysteresis, thresholds tuned on training patients only
  (subject-grouped 3-fold inner CV), same grid, **specificity floor 0.70**; unconstrained version reported alongside.
- Circular-shift surrogate test (200 shifts); per-patient Wilcoxon tests with Holm correction.

## Confirmatory hypotheses (shank/ankle controller, specificity floor 0.70)
- **H1** Timely activation exceeds the surrogate reference (one-sided p < 0.05).
- **H2** Episode detection exceeds the surrogate reference (one-sided p < 0.05).
- **H3** The controller produces fewer false alarms per hour than per-window classification
  (per-patient Wilcoxon, Holm-adjusted p < 0.05).
- **H4** Transition recall under the leakage-free protocol (P3) is markedly lower than under the
  reconstructed original protocol (P0); reported descriptively.

## Secondary (descriptive)
Comparison with HMM filtering; constrained vs unconstrained controller; lower-back sensor.

## Reporting rule
All results are reported whatever their direction. Only corrections of data-format errors that do not
use model outputs may be made after this plan is committed, and any such correction will be reported.
