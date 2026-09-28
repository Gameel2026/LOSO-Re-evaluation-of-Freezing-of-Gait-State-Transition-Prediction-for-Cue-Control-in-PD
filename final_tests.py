import pandas as pd
import fog_improve as fi

SETS = {"DAPHNET": "results_causal", "Li2021": "results_li_causal"}
rows = []
for ds, src in SETS.items():
    b3 = pd.read_csv(f"results_constrained/B3_per_subject_{ds}.csv")
    s1 = pd.read_csv(f"{src}/S1_per_subject.csv")
    table = pd.concat([b3[b3.method == "C0spec Shank"],
                       s1[s1.method.isin(["RF (per-window)", "HMM forward (causal)"])]], ignore_index=True)
    res = fi.tests(table, [("C0spec Shank", "RF (per-window)"), ("C0spec Shank", "HMM forward (causal)")])
    res.insert(0, "dataset", ds); rows.append(res)
out = pd.concat(rows, ignore_index=True).round(4)
out.to_csv("results_constrained/B5_final_tests.csv", index=False)
pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
print(out)
