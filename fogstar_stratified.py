import os
import numpy as np
import pandas as pd

import fog_improve as fi
import fog_floor_sensitivity as fs
from fog_pipeline import contiguous_runs
from fog_study import sequences

FOGSTAR_CSV = r"D:\New research fog work\FoG STAR\sensor_data.csv"
FOGSTAR_OUT = r"D:\FoG-STAR\daphnet_format"          
SRC, OUT = "results_fogstar_causal", "results_fogstar_controller"
FS_IN, FS_OUT, W, PRE_S, N_SHIFTS = 60, 64, 32, 4.0, 200
WALK = 1                                               
NAMES = {0: "unlabelled", 1: "walking", 2: "sitting", 3: "standing", 4: "sit-to-stand",
         5: "stand-to-sit", 6: "turning right", 7: "turning left"}


def run_map(df):
    out, cnt = {}, {}
    for (s, se, t), idx in df.groupby(["subjectID", "sessionID", "taskID"], sort=True).indices.items():
        if len(idx) < 2 * FS_IN:
            continue
        idx = np.sort(idx); idx = idx[np.argsort(df["timestamp"].to_numpy()[idx], kind="stable")]
        cnt[s] = cnt.get(s, 0) + 1; out[(int(s), cnt[s])] = idx
    return out


def episode_strata(meta, df, rmap):
    act = pd.to_numeric(df["activity"], errors="coerce").fillna(0).astype(int).to_numpy()
    starts, rows = {}, {}
    for (s, r) in meta[["subject", "run"]].drop_duplicates().itertuples(index=False):
        lab = np.loadtxt(os.path.join(FOGSTAR_OUT + "_ankle", f"S{int(s):02d}R{int(r):02d}.txt"))[:, -1]
        starts[(s, r)] = [a for a, b in contiguous_runs(lab != 0)]
    y2 = meta["y2"].to_numpy()
    for key, idx in sequences(meta):
        s, r, g = meta.iloc[idx[0]][["subject", "run", "seg"]]
        grp = rmap[(int(s), int(r))]
        for a, e in contiguous_runs(y2[idx] == 2):
            n_out = starts[(s, r)][int(g)] + a * W                    
            i0 = int(round(n_out * FS_IN / FS_OUT))                   
            pre = act[grp[max(0, i0 - int(PRE_S * FS_IN)):i0]]
            dom = int(pd.Series(pre).mode().iloc[0]) if len(pre) else -1
            rows[(key, a)] = {"subject": int(s), "pre_dominant": NAMES.get(dom, str(dom)),
                              "stratum": "walking-preceded" if dom == WALK else "turning/standing-preceded"}
    return rows


def per_episode(on, meta, cfg):
    y2 = meta["y2"].to_numpy(); out = {}
    for key, idx in sequences(meta):
        o = on[idx]; runs = contiguous_runs(o)
        for a, e in contiguous_runs(y2[idx] == 2):
            timely = False
            if a > 0 and o[a - 1]:
                start = next(p for p, q in runs if p < a <= q); timely = (a - start) <= cfg.pre_w
            out[(key, a)] = (timely, bool(o[a:e].any()))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(FOGSTAR_CSV, usecols=["timestamp", "activity", "subjectID", "sessionID", "taskID"])
    cfg = fi.cfg_(); fs.DATASETS = {"FoG-STAR": [SRC]}; fs.FLOORS = [0.70]
    y4, meta, on, _ = fs.run_dataset("FoG-STAR", cfg); on = on[0.70]
    summ = fi.summarise(y4, on, meta, cfg)
    print(f"\nReproduction check: timely {summ['timely_pre_onset_%']:.3f}% (pre-registered 13.861), "
          f"detection {summ['detected_%']:.3f}% (pre-registered 67.327)")
    strata = episode_strata(meta, df, run_map(df))
    obs = per_episode(on, meta, cfg)
    rng = np.random.default_rng(cfg.seed); null = []
    for _ in range(N_SHIFTS):
        sh = on.copy()
        for _, idx in sequences(meta):
            if len(idx) > 1: sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        null.append(per_episode(sh, meta, cfg))
    rows = []
    for st in ["walking-preceded", "turning/standing-preceded", "all"]:
        keys = [k for k in obs if st == "all" or strata[k]["stratum"] == st]
        if not keys: continue
        for j, name in [(0, "timely_activation_%"), (1, "detection_%")]:
            o = 100 * np.mean([obs[k][j] for k in keys]); n = np.array([100 * np.mean([d[k][j] for k in keys]) for d in null])
            rows.append({"stratum": st, "episodes": len(keys), "patients": len({strata[k]["subject"] for k in keys}),
                         "metric": name, "observed": o, "surrogate_mean": n.mean(), "surrogate_95th": np.quantile(n, 0.95),
                         "p_one_sided": (1 + (n >= o).sum()) / (1 + N_SHIFTS)})
    R = pd.DataFrame(rows).round(3); R.to_csv(f"{OUT}/X5_stratified.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print("\nPrimary controller on FoG-STAR, by dominant activity in the 4 s before onset (post hoc)\n")
    print(R.to_string(index=False))
    print("\nDominant pre-onset activity (episodes):\n", pd.Series([v["pre_dominant"] for v in strata.values()]).value_counts().to_string())
    print(f"\nDone. Saved {OUT}/X5_stratified.csv")


if __name__ == "__main__":
    main()
