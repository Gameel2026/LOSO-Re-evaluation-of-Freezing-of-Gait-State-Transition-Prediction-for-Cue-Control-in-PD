import os
import pandas as pd
import fog_rerun_causal as rc
import fog_fix2 as f2
import fog_improve as fi
import fog_floor_sensitivity as fs

FOGSTAR_OUT = r"D:\FoG-STAR\daphnet_format"          
STAGES = {"CAUSAL": True, "FIX2": True, "CONTROLLER": True}
SRC, OUT = "results_fogstar_causal", "results_fogstar_controller"

rc.ROOT, rc.OUT = FOGSTAR_OUT + "_ankle", SRC
rc.SENSOR_ROOTS = {"Shank": FOGSTAR_OUT + "_ankle", "Trunk": FOGSTAR_OUT + "_back", "Shank+waist": FOGSTAR_OUT + "_ankleback"}
rc.SENSORS = {"Shank": ("ank",), "Trunk": ("tr",), "Shank+waist": ("ank", "tr")}
f2.OUT = SRC
fs.DATASETS = {"FoG-STAR": [SRC]}


def controller():
    os.makedirs(OUT, exist_ok=True)
    cfg = fi.cfg_(); fs.FLOORS = [None, 0.70]
    y4, meta, on, ch = fs.run_dataset("FoG-STAR", cfg)
    ch.to_csv(f"{OUT}/FS_choices.csv", index=False)
    summ, sur, table = [], [], []
    for f, name in [(0.70, "Controller, floor 0.70 (primary)"), (None, "Controller, no floor")]:
        summ.append({"controller": name, **fi.summarise(y4, on[f], meta, cfg)})
        sg = fi.surrogate(y4, on[f], meta, cfg)
        for k in ["timely_pre_onset_%", "detected_%", "predicted_before_onset_%"]:
            sur.append({"controller": name, "metric": k, **sg.loc[k].to_dict()})
        table.append(fi.per_subject(name, y4, on[f], meta, cfg))
    s1 = pd.read_csv(f"{SRC}/S1_per_subject.csv")
    table = pd.concat(table + [s1[s1.method.isin(["RF (per-window)", "HMM forward (causal)"])]], ignore_index=True)
    P = "Controller, floor 0.70 (primary)"
    T = fi.tests(table, [(P, "RF (per-window)"), (P, "HMM forward (causal)"), (P, "Controller, no floor")])
    S, G = pd.DataFrame(summ), pd.DataFrame(sur)
    S.round(4).to_csv(f"{OUT}/X1_summary.csv", index=False); G.round(4).to_csv(f"{OUT}/X2_surrogate.csv", index=False)
    table.round(4).to_csv(f"{OUT}/X3_per_subject.csv", index=False); T.round(4).to_csv(f"{OUT}/X4_tests.csv", index=False)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cols = ["controller", "cue_specificity", "balanced_acc", "timely_pre_onset_%", "predicted_before_onset_%", "detected_%",
            "median_lead_time_s", "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\nController summary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate test\n", G.round(3).to_string(index=False))
    print("\nPer-patient tests (Holm within each comparison)\n", T.to_string(index=False))
    print("\nPre-registered hypotheses:")
    g = G[G.controller == P].set_index("metric")
    print(f"  H1 timely activation > surrogate : p = {g.loc['timely_pre_onset_%', 'p_one_sided']:.3f}")
    print(f"  H2 detection > surrogate         : p = {g.loc['detected_%', 'p_one_sided']:.3f}")
    fa = T[(T.comparison.str.contains("RF")) & (T.metric == "false_alarms_per_hour")].iloc[0]
    print(f"  H3 fewer false alarms than RF    : {fa.A_better} patients, Holm p = {fa.p_holm:.3f}")
    e1 = pd.read_csv(f"{SRC}/E1_protocols.csv", index_col=0)
    print(f"  H4 transition recall P0 vs P3    : {e1.iloc[0]['transition_recall']:.3f} vs {e1.iloc[-1]['transition_recall']:.3f}")


if __name__ == "__main__":
    if STAGES["CAUSAL"]:
        print("\n=== E1-E5 (causal) on FoG-STAR ==="); rc.main()
    if STAGES["FIX2"]:
        print("\n=== P0 real-only + surrogate on FoG-STAR ==="); f2.main()
    if STAGES["CONTROLLER"]:
        print("\n=== Primary controller (floor 0.70) on FoG-STAR ==="); controller()
    print("\nAll FoG-STAR analyses done.")
