import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1
import mx2_controller as mx2

SRC = "results_mx2_external"
OUT = "results_mx2_external_decomposition"
COHORTS = {"tdcsfog (lab)": "tdcsfog", "defog (home)": "defog"}
KEYS = mx2.KEYS
LATENCIES_S = [0.5, 1.0, 2.0]


def per_fold(W, C, score_col, thr):
    y2 = W["y2"].to_numpy(); s = W[score_col].to_numpy(); out = np.zeros(len(W), bool)
    for _, r in C.iterrows():
        idx = np.flatnonzero(W["subject"].to_numpy() == r["subject"])
        e = m1.Ev(W, y2, idx)
        on, off = (r["primary_on"], r["primary_off"]) if thr == "primary" else (r["M_on"], r["M_off"])
        k = 0 if thr == "primary" else int(round(r["M_hold_s"] / mx2.WIN_S))
        out[idx] = mx2.hold_k(e, e.hyst(s[idx], on, off), k)
    return out


def ideal(ev, y2, lat):
    L = int(round(lat / mx2.WIN_S)); on = np.zeros(len(y2), bool)
    starts = np.flatnonzero(ev.first); ends = np.r_[starts[1:], ev.n]
    for a, b in zip(starts, ends):
        for s, e in m1.runs(y2[a:b] == 2):
            if s + L < e:
                on[a + s + L:a + e] = True
    return on


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    T1, T2, T3, T4 = [], [], [], []
    for name, tag in COHORTS.items():
        rng = np.random.default_rng(42)
        W = pd.read_csv(os.path.join(SRC, f"EV_per_window_{tag}_seed42.csv.gz"))
        C = pd.read_csv(os.path.join(SRC, f"EV_choices_{tag}.csv"))
        y2 = W["y2"].to_numpy()
        ev = m1.Ev(W, y2, np.arange(len(W)))
        cue = {"Primary rule (registered)": W["on | Primary"].to_numpy().astype(bool),
               "M (registered)": W["on | M"].to_numpy().astype(bool),
               "Primary-rule score at M operating points": per_fold(W, C, "score | primary", "M"),
               "M score at primary-rule operating points": per_fold(W, C, "score | M", "primary")}
        chk = {k: float((cue[k] == per_fold(W, C, "score | primary" if k.startswith("Primary") else "score | M",
                                             "primary" if k.startswith("Primary") else "M")).mean()) for k in list(cue)[:2]}
        print(f"{name}: registered cue states reproduced, agreement {chk}", flush=True)
        cue["Continuous cueing (always on)"] = np.ones(len(W), bool)
        for lat in LATENCIES_S:
            cue[f"Ideal detector, {lat:g} s latency"] = ideal(ev, y2, lat)
        cnt = {k: ev.counts(v) for k, v in cue.items()}
        for k, c in cnt.items():
            ci = m1.boot_ci(c, rng, KEYS)
            T1.append({"cohort": name, "strategy": k, **{m: ci[m][0] for m in KEYS},
                       **{f"{m}_CI_low": ci[m][1] for m in KEYS}, **{f"{m}_CI_high": ci[m][2] for m in KEYS}})
        steps = [("Order A: threshold selection", "Primary-rule score at M operating points", "Primary rule (registered)"),
                 ("Order A: score (larger ensemble)", "M (registered)", "Primary-rule score at M operating points"),
                 ("Order B: score (larger ensemble)", "M score at primary-rule operating points", "Primary rule (registered)"),
                 ("Order B: threshold selection", "M (registered)", "M score at primary-rule operating points"),
                 ("Total: M minus primary rule", "M (registered)", "Primary rule (registered)")]
        for lab, a, b in steps:
            d = m1.boot_diff(cnt[a], cnt[b], rng, KEYS)
            for m in KEYS:
                T2.append({"cohort": name, "component": lab, "metric": m, "difference": d[m][0], "CI_low": d[m][1], "CI_high": d[m][2]})
        h = len(W) * mx2.WIN_S / 3600
        T3.append({"cohort": name, "patients": W["subject"].nunique(), "patients_with_FoG": int((cnt["M (registered)"].episodes > 0).sum()),
                   "episodes": int(cnt["M (registered)"].episodes.sum()), "hours": h, "time_in_FoG_%": 100 * (y2 == 2).mean(),
                   "time_in_episode_zone_%": 100 * ev.zone.mean()})
        a = cnt["M (registered)"].set_index("subject"); b = cnt["Primary rule (registered)"].set_index("subject").reindex(a.index)
        hrs = a["windows"] * mx2.WIN_S / 3600
        dfa = (a["false_alarms"] - b["false_alarms"]) / hrs
        ep = a["episodes"] > 0
        dcov = 100 * (a["detected"] - b["detected"])[ep] / a["episodes"][ep]
        T4.append({"cohort": name, "patients": len(a), "fewer_false_alarms": int((dfa < 0).sum()), "equal_false_alarms": int((dfa == 0).sum()),
                   "more_false_alarms": int((dfa > 0).sum()), "median_FA_difference_per_h": float(dfa.median()),
                   "FA_difference_IQR_low": float(dfa.quantile(0.25)), "FA_difference_IQR_high": float(dfa.quantile(0.75)),
                   "patients_with_episodes": int(ep.sum()), "lower_coverage": int((dcov < 0).sum()), "equal_coverage": int((dcov == 0).sum()),
                   "higher_coverage": int((dcov > 0).sum()), "median_coverage_difference": float(dcov.median())})
        print(f"{name}: done", flush=True)
    for T, f in [(T1, "EVD_T1_reference_strategies.csv"), (T2, "EVD_T2_decomposition.csv"), (T3, "EVD_T3_cohort_description.csv"), (T4, "EVD_T4_per_patient.csv")]:
        pd.DataFrame(T).round(3).to_csv(os.path.join(OUT, f), index=False)
    print("\n", pd.DataFrame(T1)[["cohort", "strategy"] + KEYS].round(1).to_string(index=False))
    D = pd.DataFrame(T2); print("\n", D[D.metric.isin(["false_alarms_per_hour", "coverage_%"])].round(2).to_string(index=False))
    print("\n", pd.DataFrame(T3).round(2).to_string(index=False))
    print("\n", pd.DataFrame(T4).round(2).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
