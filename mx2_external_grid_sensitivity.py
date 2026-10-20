import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1
import mx2_controller as mx2

SRC = "results_mx2_external"
OUT = "results_mx2_external_grid"
COHORTS = {"tdcsfog (lab)": "tdcsfog", "defog (home)": "defog"}
ON = [round(a, 2) for a in np.arange(0.10, 0.91, 0.05)]
OFF = [0.05, 0.10, 0.20]
POINTS = [(a, b) for a in ON for b in OFF if b < a]
KEYS = mx2.KEYS
SCORE = {"Primary rule": "score | primary", "M": "score | M"}
FLAG = {"Primary rule": "on | Primary", "M": "on | M"}


def load(tag):
    W = pd.read_csv(os.path.join(SRC, f"EV_per_window_{tag}_seed42.csv.gz"))
    C = pd.read_csv(os.path.join(SRC, f"EV_choices_{tag}.csv"))
    return W, C


def check(W, C):
    ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
    rec = {k: np.zeros(len(W), bool) for k in SCORE}
    for _, r in C.iterrows():
        idx = np.flatnonzero(W["subject"].to_numpy() == r["subject"])
        e = m1.Ev(W, W["y2"].to_numpy(), idx)
        rec["Primary rule"][idx] = e.hyst(W[SCORE["Primary rule"]].to_numpy()[idx], r["primary_on"], r["primary_off"])
        k = int(round(r["M_hold_s"] / mx2.WIN_S))
        rec["M"][idx] = mx2.hold_k(e, e.hyst(W[SCORE["M"]].to_numpy()[idx], r["M_on"], r["M_off"]), k)
    return ev, {k: float((rec[k] == W[FLAG[k]].to_numpy().astype(bool)).mean()) for k in SCORE}


def sweep(name, W, ev, rng):
    rows, diffs = [], []
    for on, off in POINTS:
        c = {k: ev.counts(ev.hyst(W[SCORE[k]].to_numpy(), on, off)) for k in SCORE}
        for k in SCORE:
            rows.append({"cohort": name, "on": on, "off": off, "controller": k, **m1.pooled(c[k])})
        d = m1.boot_diff(c["M"], c["Primary rule"], rng, KEYS)
        for k in KEYS:
            diffs.append({"cohort": name, "on": on, "off": off, "metric": k, "difference": d[k][0], "CI_low": d[k][1], "CI_high": d[k][2]})
    return rows, diffs


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(42)
    T1, T2, chk = [], [], []
    for name, tag in COHORTS.items():
        W, C = load(tag)
        ev, agree = check(W, C)
        chk.append({"cohort": name, **{f"agreement | {k}": v for k, v in agree.items()}})
        print(f"{name}: per-fold operating points reproduced, agreement {agree}", flush=True)
        r, d = sweep(name, W, ev, rng)
        T1 += r; T2 += d
    pd.DataFrame(chk).to_csv(os.path.join(OUT, "EVG_T0_reproduction.csv"), index=False)
    A = pd.DataFrame(T1).round(3); D = pd.DataFrame(T2).round(3)
    A.to_csv(os.path.join(OUT, "EVG_T1_operating_points.csv"), index=False)
    D.to_csv(os.path.join(OUT, "EVG_T2_paired_differences.csv"), index=False)
    show = A[(A.on <= 0.40) & (A.off <= 0.10)][["cohort", "on", "off", "controller"] + KEYS]
    print("\nOperating points at and below the grid bound\n", show.round(1).to_string(index=False))
    fd = D[(D.metric.isin(["false_alarms_per_hour", "coverage_%"])) & (D.on <= 0.40) & (D.off <= 0.10)]
    print("\nM minus primary rule at and below the grid bound\n", fd.round(2).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
