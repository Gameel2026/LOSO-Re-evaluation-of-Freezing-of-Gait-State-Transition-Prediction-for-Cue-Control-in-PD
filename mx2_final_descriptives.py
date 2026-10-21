import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

OUT = "results_final_descriptives"
FILES = {"DAPHNET": "results_mx2/MX2_per_window_DAPHNET_seed42.csv.gz",
         "Multimodal FoG": "results_mx2/MX2_per_window_Multimodal_FoG_seed42.csv.gz",
         "FoG-STAR": "results_mx2/MX2_per_window_FoG-STAR_seed42.csv.gz",
         "tdcsfog (lab)": "results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz",
         "defog (home)": "results_mx2_external/EV_per_window_defog_seed42.csv.gz"}
EXTERNAL = ["tdcsfog (lab)", "defog (home)"]
CONTROLLERS = {"Primary rule": "on | Primary", "M": "on | M"}


def coverage_surrogate(ev, cue, rng):
    obs = m1.pooled(ev.counts(cue))["coverage_%"]
    starts = np.flatnonzero(ev.first); ends = np.r_[starts[1:], ev.n]; null = []
    for _ in range(m1.N_SHIFT):
        sh = cue.copy()
        for a, b in zip(starts, ends):
            if b - a > 1:
                sh[a:b] = np.roll(cue[a:b], rng.integers(1, b - a))
        null.append(m1.pooled(ev.counts(sh))["coverage_%"])
    null = np.array(null)
    return {"observed": obs, "surrogate_mean": null.mean(), "surrogate_95th": np.quantile(null, 0.95),
            "p_one_sided": (1 + (null >= obs).sum()) / (1 + m1.N_SHIFT)}


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    T1, T2, T3 = [], [], []
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"{name}: {f} not found - skipped", flush=True); continue
        W = pd.read_csv(f)
        ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
        dur = (ev.ee - ev.es) * m1.WIN_S
        T1.append({"cohort": name, "episodes": len(dur), "duration_median_s": np.median(dur),
                   "duration_p25_s": np.quantile(dur, 0.25), "duration_p75_s": np.quantile(dur, 0.75),
                   "episodes_up_to_1s_%": 100 * (dur <= 1.0).mean(), "episodes_up_to_2s_%": 100 * (dur <= 2.0).mean(),
                   "episodes_up_to_5s_%": 100 * (dur <= 5.0).mean()})
        for lab, col in CONTROLLERS.items():
            c = ev.counts(W[col].to_numpy().astype(bool))
            h_all = c["windows"].sum() * m1.WIN_S / 3600; h_free = c["outside_windows"].sum() * m1.WIN_S / 3600
            T2.append({"cohort": name, "controller": lab, "hours_analysed": h_all, "hours_outside_episode_zone": h_free,
                       "false_alarms": int(c["false_alarms"].sum()), "false_alarms_per_hour": c["false_alarms"].sum() / h_all,
                       "false_alarms_per_hour_outside_episode_zone": c["false_alarms"].sum() / h_free})
            if name in EXTERNAL:
                r = coverage_surrogate(ev, W[col].to_numpy().astype(bool), np.random.default_rng(42))
                T3.append({"cohort": name, "controller": lab, "metric": "coverage_%", **r})
        print(f"{name}: done", flush=True)
    for T, fn in [(T1, "FD_T1_episode_durations.csv"), (T2, "FD_T2_false_alarms_time_base.csv"), (T3, "FD_T3_coverage_surrogate.csv")]:
        pd.DataFrame(T).round(3).to_csv(os.path.join(OUT, fn), index=False)
        print("\n", pd.DataFrame(T).round(3).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
