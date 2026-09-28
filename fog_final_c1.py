import numpy as np
import pandas as pd
import fog_improve as fi
from fog_pipeline import relabel_four_class

RUN = {"SENSORS": True, "SURROGATE": True, "STATS": True}
SENSORS_C1 = [("Thigh", "Thigh"), ("Trunk", "Trunk"), ("All three", "All_three")]


def main():
    cfg = fi.cfg_(); OUT = fi.OUT
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    persub = [pd.read_csv(f"{OUT}/I3_per_subject.csv")]

    if RUN["SURROGATE"]:
        X0, meta = fi.load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
        on, _ = fi.controller(fi.add_context(X0, meta), y4, meta, cfg, ("RF",), "window_f1", "C1 Shank")
        sur = fi.surrogate(y4, on, meta, cfg).round(4); sur.to_csv(f"{OUT}/J2_surrogate_C1_shank.csv"); print(sur)

    if RUN["SENSORS"]:
        rows = {}
        for name, cache in SENSORS_C1:
            print(f"\nC1 {name} ...")
            Xs, ms = fi.load(cache); ys = relabel_four_class(ms, cfg.pre_w, cfg.post_w)
            on, ch = fi.controller(fi.add_context(Xs, ms), ys, ms, cfg, ("RF",), "window_f1", f"C1 {name}")
            ch.to_csv(f"{OUT}/J_choices_C1_{cache}.csv")
            rows[name] = fi.summarise(ys, on, ms, cfg); pd.DataFrame(rows).T.round(3).to_csv(f"{OUT}/J1_sensors_C1.csv")
            persub.append(fi.per_subject(f"C1 {name}", ys, on, ms, cfg))
            pd.concat(persub).round(3).to_csv(f"{OUT}/J3_per_subject.csv", index=False)
        print("\nSensors (C1)\n", pd.DataFrame(rows).T.round(3))

    if RUN["STATS"]:
        t = pd.read_csv(f"{OUT}/J3_per_subject.csv")
        old = pd.read_csv(f"{fi.SRC}/S1_per_subject.csv")
        old = old[old.method.isin(["RF (per-window)", "HMM forward (causal)"])]
        table = pd.concat([t, old], ignore_index=True)
        comps = [("C1 Shank", "C0 Shank"), ("C1 Shank", "RF (per-window)"),
                 ("C1 Shank", "HMM forward (causal)"), ("C1 Trunk", "C1 Shank")]
        res = fi.tests(table, comps).round(4); res.to_csv(f"{OUT}/J4_wilcoxon_C1.csv", index=False); print(res)
    print("\nDone.")


if __name__ == "__main__":
    main()
