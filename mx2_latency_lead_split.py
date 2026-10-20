import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

OUT = "results_latency_lead_split"
FILES = {"DAPHNET": "results_mx2/MX2_per_window_DAPHNET_seed42.csv.gz",
         "Multimodal FoG": "results_mx2/MX2_per_window_Multimodal_FoG_seed42.csv.gz",
         "FoG-STAR": "results_mx2/MX2_per_window_FoG-STAR_seed42.csv.gz",
         "tdcsfog (lab)": "results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz",
         "defog (home)": "results_mx2_external/EV_per_window_defog_seed42.csv.gz"}
CONTROLLERS = {"Primary rule": "on | Primary", "M": "on | M"}


def episode_rows(ev, cue):
    prev = np.r_[False, cue[:-1]] & ~ev.first
    rs = cue & ~prev
    rstart = np.maximum.accumulate(np.where(rs, np.arange(ev.n), -1))
    has_pre = ev.es > ev.ea
    pm1 = np.where(has_pre, ev.es - 1, 0)
    pre = has_pre & cue[pm1]
    st = rstart[pm1]
    lead = ev.es - st
    lead2 = pre & (lead >= 2) & (lead <= m1.PRE_W)
    carried = lead2 & (ev.ps >= 0) & (st >= ev.ps) & (st < ev.pe + m1.POST_W)
    rows = []
    for k in range(len(ev.es)):
        seg = cue[ev.es[k]:ev.ee[k]]
        covered = bool(seg.any())
        if pre[k]:
            lat = 0.0
        elif covered:
            lat = float(np.argmax(seg)) * m1.WIN_S
        else:
            lat = np.nan
        rows.append({"subject": ev.esubj[k], "covered": covered, "on_at_onset": bool(pre[k]), "latency_s": lat,
                     "lead2": bool(lead2[k]), "lead2_carried": bool(carried[k])})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    T = []
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"{name}: {f} not found - skipped", flush=True); continue
        W = pd.read_csv(f)
        ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
        for lab, col in CONTROLLERS.items():
            cue = W[col].to_numpy().astype(bool)
            check = m1.pooled(ev.counts(cue))
            E = episode_rows(ev, cue)
            cov = E[E.covered]
            T.append({"cohort": name, "controller": lab, "episodes": len(E), "coverage_%": 100 * E.covered.mean(),
                      "coverage_check_%": check["coverage_%"], "on_at_onset_%_of_episodes": 100 * E.on_at_onset.mean(),
                      "latency_median_s_covered": cov.latency_s.median(), "latency_p75_s_covered": cov.latency_s.quantile(0.75),
                      "latency_p90_s_covered": cov.latency_s.quantile(0.90),
                      "lead2_%": 100 * E.lead2.mean(), "lead2_check_%": check["timely_lead1_%"],
                      "lead2_new_cue_%": 100 * (E.lead2 & ~E.lead2_carried).mean(), "lead2_carried_over_%": 100 * E.lead2_carried.mean()})
        print(f"{name}: done", flush=True)
    R = pd.DataFrame(T).round(3)
    R.to_csv(os.path.join(OUT, "LAT_T1_latency_lead_split.csv"), index=False)
    print("\n", R.to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
