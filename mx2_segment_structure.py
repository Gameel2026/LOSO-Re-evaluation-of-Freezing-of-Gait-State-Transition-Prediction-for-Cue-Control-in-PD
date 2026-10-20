import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

OUT = "results_segment_structure"
FILES = {"DAPHNET": "results_mx2/MX2_per_window_DAPHNET_seed42.csv.gz",
         "Multimodal FoG": "results_mx2/MX2_per_window_Multimodal_FoG_seed42.csv.gz",
         "FoG-STAR": "results_mx2/MX2_per_window_FoG-STAR_seed42.csv.gz",
         "tdcsfog (lab)": "results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz",
         "defog (home)": "results_mx2_external/EV_per_window_defog_seed42.csv.gz"}
MIN_LEN = [60, 120]


def describe(name, W):
    key = ["subject", "run", "seg"]
    L = W.groupby(key).size()
    on_idx = []
    for _, v in W.groupby(key, sort=False):
        y = (v["y2"].to_numpy() == 2).astype(int)
        e = np.flatnonzero(np.diff(np.r_[0, y]))
        on_idx += [i for i in e if y[i] == 1]
    on_idx = np.array(on_idx)
    fog_seg = W.groupby(key)["y2"].apply(lambda v: (v == 2).any())
    return {"cohort": name, "recordings": W.groupby(["subject", "run"]).ngroups, "segments": len(L),
            "segments_with_FoG": int(fog_seg.sum()), "segment_length_median_s": L.median() * m1.WIN_S,
            "segment_length_p10_s": L.quantile(0.10) * m1.WIN_S, "segment_length_p90_s": L.quantile(0.90) * m1.WIN_S,
            "episodes": len(on_idx), "onsets_within_4s_of_segment_start_%": 100 * (on_idx < m1.PRE_W).mean(),
            "onsets_at_segment_start_%": 100 * (on_idx == 0).mean()}


def restricted(name, W):
    rows = []
    key = ["subject", "run", "seg"]
    L = W.groupby(key)["y2"].transform("size").to_numpy()
    for k in MIN_LEN:
        idx = np.flatnonzero(L >= k)
        ev = m1.Ev(W, W["y2"].to_numpy(), idx)
        cue = W["on | M"].to_numpy().astype(bool)[idx]
        p = m1.pooled(ev.counts(cue))
        for r in m1.surrogate(ev, cue, np.random.default_rng(42)):
            rows.append({"cohort": name, "min_segment_length_s": k * m1.WIN_S, "windows_retained_%": 100 * len(idx) / len(W),
                         "episodes_retained": p["episodes"], **r})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    T1, T2 = [], []
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"{name}: {f} not found - skipped", flush=True); continue
        W = pd.read_csv(f)
        T1.append(describe(name, W))
        if name in ("tdcsfog (lab)", "defog (home)"):
            T2 += restricted(name, W)
        print(f"{name}: done", flush=True)
    A = pd.DataFrame(T1).round(2); B = pd.DataFrame(T2).round(3)
    A.to_csv(os.path.join(OUT, "SEG_T1_segment_structure.csv"), index=False)
    B.to_csv(os.path.join(OUT, "SEG_T2_timing_long_segments.csv"), index=False)
    print("\n", A.to_string(index=False))
    print("\n", B.to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
