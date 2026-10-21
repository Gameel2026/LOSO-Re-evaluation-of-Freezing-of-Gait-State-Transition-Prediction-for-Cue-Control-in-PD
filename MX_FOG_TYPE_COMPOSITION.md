# Descriptive: composition of annotated FoG types in the external cohorts

Status: descriptive characterisation of the external datasets, written after all registered and post hoc results were available and committed together with its script before the script was executed. Only the original annotation columns are read; no model output, cue state or result file is used, and no hypothesis is stated. EV1 to EV4 and all other results are not changed. All outputs are reported.

## Purpose

The conversion merged the annotated FoG types into one FoG label. The share of each annotated type (StartHesitation, Turn, Walking; Unspecified for the defog recordings without type) is reported for each external cohort, as context for the dataset description.

## Procedure

Every recording in `kaggle_subject_mapping.csv` is converted again with `kaggle_convert.convert_file` (identical rules, including the 2-s merging and the Valid and Task masks). Each FoG episode of the converted label receives the majority type of its annotated samples at 64 Hz (nearest-sample rule of the conversion); samples added by the 2-s merging carry no type and are ignored. Per cohort and type, the number and share of episodes, the number of patients, the FoG time and its share, the median episode duration and the number of episodes containing more than one type are reported. The episode count per cohort is compared with the registered counts (tdcsfog 949, defog 689).

Script: `mx2_fog_type_composition.py`. Outputs (`results_fog_type_composition/`): `FTC_T0_episode_check.csv`, `FTC_T1_type_composition.csv`, `FTC_T2_episodes.csv`.
