import os
import pandas as pd
import fog_improve as fi
from fog_pipeline import relabel_four_class

OUT = "results_constrained"
RUN_SURROGATE = True
SETS = {"DAPHNET": ("results_causal", [("Thigh", "Thigh"), ("Trunk", "Trunk"), ("All three", "All_three")]),
        "Li2021": ("results_li_causal", [("Trunk", "Trunk"), ("Shank+waist", "Shank+waist")])}


def main():
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cfg = fi.cfg_()
    b1 = pd.read_csv(f"{OUT}/B1_summary.csv", index_col=0)
    summary = {f"{ds} Shank": b1.loc[f"{ds} C0spec"] for ds in SETS}
    persub, tests = [], []
    for ds, (src, sensors) in SETS.items():
        fi.SRC = src
        b3 = pd.read_csv(f"{OUT}/B3_per_subject_{ds}.csv")
        shank = b3[b3.method == "C0spec Shank"].copy()
        ds_rows = [shank]
        for name, cache in sensors:
            print(f"\n=== {ds}: {name} (final controller) ===")
            X, meta = fi.load(cache); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
            on, ch = fi.controller(X, y4, meta, cfg, ("RF",), "f1_spec", f"{ds} {name}")
            ch.to_csv(f"{OUT}/S_choices_{ds}_{cache}.csv")
            print(f"  constraint changed the F1 choice in {int(ch.constraint_changed_choice.sum())}/{len(ch)} folds")
            summary[f"{ds} {name}"] = pd.Series(fi.summarise(y4, on, meta, cfg))
            pd.DataFrame(summary).T.round(3).to_csv(f"{OUT}/S_sensors_summary.csv")
            ps = fi.per_subject(f"C0spec {name}", y4, on, meta, cfg); ds_rows.append(ps)
            if RUN_SURROGATE:
                sur = fi.surrogate(y4, on, meta, cfg).round(4)
                sur.to_csv(f"{OUT}/S_surrogate_{ds}_{cache}.csv"); print(sur)
        table = pd.concat(ds_rows, ignore_index=True); table.insert(0, "dataset", ds); persub.append(table)
        pd.concat(persub).round(3).to_csv(f"{OUT}/S_per_subject_sensors.csv", index=False)
        comps = [(f"C0spec {n}", "C0spec Shank") for n, _ in sensors]
        res = fi.tests(table.drop(columns="dataset"), comps).round(4); res.insert(0, "dataset", ds); tests.append(res)
        pd.concat(tests).to_csv(f"{OUT}/S_tests_sensors.csv", index=False)
        print(f"\n{ds} tests (vs shank)\n", res)
    print("\nSensor summary (final controller)\n", pd.DataFrame(summary).T.round(3))
    print(f"\nDone. Files S_* in {OUT}/")


if __name__ == "__main__":
    main()
