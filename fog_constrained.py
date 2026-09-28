import os
import pandas as pd
import fog_improve as fi
from fog_pipeline import relabel_four_class

DATASETS = {"DAPHNET": ("results_causal", "results_improve"),
            "Li2021": ("results_li_causal", "results_li_improve")}
OBJECTIVE = "f1_spec"                             
STEPS = {"C0spec": False, "C1spec": True}        
OUT = "results_constrained"
RUN_SURROGATE = True


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cfg = fi.cfg_(); summary = {}
    for ds, (src, imp) in DATASETS.items():
        fi.SRC = src
        X0, meta = fi.load("Shank"); y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
        persub = []
        for step, ctx in STEPS.items():
            print(f"\n=== {ds}: {step} ({OBJECTIVE}, SPEC_MIN={fi.SPEC_MIN}) ===")
            X = fi.add_context(X0, meta) if ctx else X0
            on, ch = fi.controller(X, y4, meta, cfg, ("RF",), OBJECTIVE, f"{ds} {step}")
            print(f"  constraint changed the F1 choice in {int(ch.constraint_changed_choice.sum())}/{len(ch)} folds; "
                  f"infeasible folds: {int((~ch.constraint_feasible.astype(bool)).sum())}")
            ch.to_csv(f"{OUT}/B_choices_{ds}_{step}.csv")
            summary[f"{ds} {step}"] = fi.summarise(y4, on, meta, cfg)
            pd.DataFrame(summary).T.round(3).to_csv(f"{OUT}/B1_summary.csv")
            persub.append(fi.per_subject(f"{step} Shank", y4, on, meta, cfg))
            if RUN_SURROGATE:
                sur = fi.surrogate(y4, on, meta, cfg).round(4)
                sur.to_csv(f"{OUT}/B2_surrogate_{ds}_{step}.csv"); print(sur)
        ref = pd.read_csv(f"{imp}/J3_per_subject.csv") if os.path.exists(f"{imp}/J3_per_subject.csv") \
            else pd.read_csv(f"{imp}/I3_per_subject.csv")
        old = pd.read_csv(f"{src}/S1_per_subject.csv"); old = old[old.method == "RF (per-window)"]
        table = pd.concat(persub + [ref[ref.method.isin(["C0 Shank", "C1 Shank"])], old], ignore_index=True)
        table.round(3).to_csv(f"{OUT}/B3_per_subject_{ds}.csv", index=False)
        comps = [("C0spec Shank", "C0 Shank"), ("C1spec Shank", "C1 Shank"),
                 ("C1spec Shank", "RF (per-window)"), ("C1spec Shank", "C0spec Shank")]
        res = fi.tests(table, comps).round(4); res.to_csv(f"{OUT}/B4_tests_{ds}.csv", index=False)
        print(f"\n{ds} tests\n", res)
    print("\nSummary\n", pd.DataFrame(summary).T.round(3))
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
