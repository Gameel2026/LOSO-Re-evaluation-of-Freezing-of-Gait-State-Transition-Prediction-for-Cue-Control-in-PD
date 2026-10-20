import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

SRC = "results_mx2_external"
OUT = "results_mx2_external_operating_surrogate"
COHORTS = {"tdcsfog (lab)": "tdcsfog", "defog (home)": "defog"}
POINTS = [(0.20, 0.05), (0.20, 0.10), (0.25, 0.05), (0.25, 0.10), (0.30, 0.05), (0.30, 0.10)]
SCORE = "score | M"


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rows = []
    for name, tag in COHORTS.items():
        W = pd.read_csv(os.path.join(SRC, f"EV_per_window_{tag}_seed42.csv.gz"))
        ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
        cues = {"registered per-fold operating points": W["on | M"].to_numpy().astype(bool)}
        for on, off in POINTS:
            cues[f"fixed on {on:.2f} / off {off:.2f}"] = ev.hyst(W[SCORE].to_numpy(), on, off)
        for label, cue in cues.items():
            p = m1.pooled(ev.counts(cue))
            for r in m1.surrogate(ev, cue, np.random.default_rng(42)):
                rows.append({"cohort": name, "operating_point": label, "false_alarms_per_hour": p["false_alarms_per_hour"],
                             "time_cue_on_%": p["time_cue_on_%"], "coverage_%": p["coverage_%"], **r})
            print(f"{name}, {label}: done", flush=True)
    R = pd.DataFrame(rows).round(3)
    R.to_csv(os.path.join(OUT, "EVOS_T1_operating_point_surrogate.csv"), index=False)
    print("\n", R.to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
