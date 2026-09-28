import os
import traceback
import pandas as pd
import run_li                      
import fog_improve as fi
import fog_final_c1 as fc
from fog_pipeline import relabel_four_class

DO = {"C1_FINAL": True, "C3": True}


def run_c3():
    cfg = fi.cfg_(); OUT = fi.OUT
    X0, meta = fi.load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    on, ch = fi.controller(fi.add_context(X0, meta), y4, meta, cfg, ("RF", "GBoost"), "event_f1", "C3 +model selection")
    ch.to_csv(f"{OUT}/I_choices_C3_Shank.csv")
    i1 = pd.read_csv(f"{OUT}/I1_ablation_shank.csv", index_col=0)
    i1.loc["C3 +model selection"] = pd.Series(fi.summarise(y4, on, meta, cfg))
    i1.round(3).to_csv(f"{OUT}/I1_ablation_shank.csv"); print("\nAblation\n", i1.round(3))
    i3 = pd.read_csv(f"{OUT}/I3_per_subject.csv")
    i3 = pd.concat([i3[i3.method != "C3 Shank"], fi.per_subject("C3 Shank", y4, on, meta, cfg)], ignore_index=True)
    i3.round(3).to_csv(f"{OUT}/I3_per_subject.csv", index=False)
    sur = fi.surrogate(y4, on, meta, cfg).round(4); sur.to_csv(f"{OUT}/I5_surrogate_C3_shank.csv"); print(sur)
    old = pd.read_csv(f"{fi.SRC}/S1_per_subject.csv")
    table = pd.concat([i3, old[old.method.isin(["RF (per-window)", "HMM forward (causal)"])]], ignore_index=True)
    comps = [("C3 Shank", "C0 Shank"), ("C3 Shank", "RF (per-window)"), ("C3 Shank", "HMM forward (causal)")]
    res = fi.tests(table, comps).round(4); res.to_csv(f"{OUT}/I4_wilcoxon.csv", index=False); print(res)


if __name__ == "__main__":
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    print("Paths:", fi.SRC, "->", fi.OUT)
    if DO["C1_FINAL"]:
        print("\n=== C1 final stage on Li 2021 ===")
        fc.SENSORS_C1 = [("Trunk", "Trunk"), ("Shank+waist", "Shank+waist")]
        fc.RUN = {"SENSORS": True, "SURROGATE": True, "STATS": True}
        fc.main()
    if DO["C3"]:
        print("\n=== C3 (shank) on Li 2021 ===")
        try:
            run_c3()
        except Exception:
            print("\nC3 FAILED - everything above is saved. Send this traceback:\n")
            traceback.print_exc()
    print("\nResume finished.")
