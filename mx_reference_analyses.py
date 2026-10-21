import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [os.path.join(HERE, "results_methodsx"), HERE, os.path.join(os.path.dirname(HERE), "results_methodsx")]
SRC = next((c for c in CANDIDATES if os.path.exists(os.path.join(c, "MX_per_window.csv.gz"))), None)
if SRC is None:
    sys.exit("MX_per_window.csv.gz not found next to this script or in results_methodsx/")
OUT = os.path.join(SRC, "analyses")

WIN_S = 0.5
PRE_W = 8
POST_W = 6
N_BOOT = 5000
N_SHIFT = 200
SEED = 42
PRIMARY = "Controller, floor 0.70"
LATENCIES_S = [0.5, 1.0, 2.0]
FA_TARGETS = [25, 50, 100, 200]
ON_TARGETS = [20, 30, 40, 50]


def runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


def stretches(df):
    out = []
    for key, g in df.groupby(["subject", "run", "seg"], sort=True):
        out.append((key[0], g.sort_values("pos").index.to_numpy()))
    return out


def hysteresis(score, on_thr, off_thr):
    out = np.zeros(len(score), bool)
    state = False
    for t in range(len(score)):
        s = score[t]
        state = (s >= on_thr) if not state else (s >= off_thr)
        out[t] = state
    return out


def apply_hyst(score, st, on_thr, off_thr):
    on = np.zeros(len(score), bool)
    for _, idx in st:
        on[idx] = hysteresis(score[idx], on_thr, off_thr)
    return on


def episode_table(y2, on, st, pre_w=PRE_W, post_w=POST_W, min_lead=0):
    rows = {}
    for subj, idx in st:
        r = rows.setdefault(subj, {"episodes": 0, "detected": 0, "pre_onset": 0, "timely": 0, "early": 0,
                                   "after_onset": 0, "missed": 0, "false_alarms": 0, "windows": 0, "on_windows": 0})
        fog = y2[idx] == 2
        o = on[idx]
        r["windows"] += len(idx); r["on_windows"] += int(o.sum())
        eps = runs(fog); ons = runs(o)
        zone = np.zeros(len(idx), bool)
        for s, e in eps:
            zone[max(0, s - pre_w):min(len(idx), e + post_w)] = True
        r["false_alarms"] += sum(1 for a, b in ons if not zone[a:b].any())
        for s, e in eps:
            r["episodes"] += 1
            det = bool(o[s:e].any())
            r["detected"] += det
            if s > 0 and o[s - 1]:
                r["pre_onset"] += 1
                start = next(a for a, b in ons if a < s <= b)
                lead = s - start
                if min_lead <= lead <= PRE_W:
                    r["timely"] += 1
                else:
                    r["early"] += 1
            elif det:
                r["after_onset"] += 1
            else:
                r["missed"] += 1
    df = pd.DataFrame(rows).T
    df.index.name = "subject"
    df["hours"] = df.windows * WIN_S / 3600
    return df.reset_index()


def pooled(c):
    e = max(c.episodes.sum(), 1)
    return {"episodes": int(c.episodes.sum()),
            "coverage_%": 100 * c.detected.sum() / e,
            "timely_activation_%": 100 * c.timely.sum() / e,
            "false_alarms_per_hour": c.false_alarms.sum() / c.hours.sum(),
            "time_cue_on_%": 100 * c.on_windows.sum() / c.windows.sum()}


def boot(c, rng):
    c = c.reset_index(drop=True)
    est = pooled(c)
    keys = ["coverage_%", "timely_activation_%", "false_alarms_per_hour", "time_cue_on_%"]
    B = {k: [] for k in keys}
    n = len(c)
    for _ in range(N_BOOT):
        s = c.iloc[rng.integers(0, n, n)]
        if s.episodes.sum() == 0:
            continue
        p = pooled(s)
        for k in keys:
            B[k].append(p[k])
    for k in keys:
        lo, hi = np.percentile(B[k], [2.5, 97.5])
        est[k + "_CI_low"] = lo; est[k + "_CI_high"] = hi
    return est


def strategies(d, st):
    y2 = d.y2.to_numpy()
    out = {c.replace("on | ", ""): d[c].to_numpy().astype(bool) for c in d.columns if c.startswith("on | ")}
    out["Continuous cueing (always on)"] = np.ones(len(d), bool)
    for lat in LATENCIES_S:
        L = int(round(lat / WIN_S)); on = np.zeros(len(d), bool)
        for _, idx in st:
            for s, e in runs(y2[idx] == 2):
                if s + L < e:
                    on[idx[s + L:e]] = True
        out[f"Ideal detector, {lat:g} s latency"] = on
    return out


def curve_points(d, st, score, family, hyst):
    y2 = d.y2.to_numpy(); pts = []
    if hyst:
        qs = np.quantile(score, np.linspace(0.30, 0.98, 18))
        pairs = [(a, b) for i, a in enumerate(qs) for b in qs[:i + 1][::3]]
    else:
        pairs = [(a, a) for a in np.quantile(score, np.linspace(0.20, 0.995, 40))]
    for a, b in pairs:
        on = apply_hyst(score, st, a, b)
        p = pooled(episode_table(y2, on, st))
        pts.append({"family": family, "theta_on": a, "theta_off": b, **p})
    return pts


def frontier(P, xcol, targets, ycol="timely_activation_%"):
    rows = []
    for fam, g in P.groupby("family"):
        for t in targets:
            ok = g[g[xcol] <= t]
            best = ok.sort_values(ycol, ascending=False).iloc[0] if len(ok) else None
            rows.append({"family": fam, f"{xcol}_max": t,
                         ycol: best[ycol] if best is not None else np.nan,
                         "coverage_%": best["coverage_%"] if best is not None else np.nan,
                         "false_alarms_per_hour": best["false_alarms_per_hour"] if best is not None else np.nan,
                         "time_cue_on_%": best["time_cue_on_%"] if best is not None else np.nan})
    return pd.DataFrame(rows)


def patient_surrogate(d, st, on, rng):
    y2 = d.y2.to_numpy(); rows = []
    obs = episode_table(y2, on, st).set_index("subject")
    null = {s: [] for s in obs.index}
    for _ in range(N_SHIFT):
        sh = on.copy()
        for _, idx in st:
            if len(idx) > 1:
                sh[idx] = np.roll(on[idx], rng.integers(1, len(idx)))
        t = episode_table(y2, sh, st).set_index("subject")
        for s in obs.index:
            null[s].append(100 * t.loc[s, "timely"] / max(t.loc[s, "episodes"], 1))
    for s in obs.index:
        if obs.loc[s, "episodes"] == 0:
            continue
        o = 100 * obs.loc[s, "timely"] / obs.loc[s, "episodes"]; n = np.array(null[s])
        rows.append({"subject": s, "episodes": int(obs.loc[s, "episodes"]), "observed_timely_%": o,
                     "surrogate_mean_%": n.mean(), "surrogate_95th_%": np.quantile(n, 0.95),
                     "p_one_sided": (1 + (n >= o).sum()) / (1 + N_SHIFT)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(SEED)
    D = pd.read_csv(os.path.join(SRC, "MX_per_window.csv.gz"))
    T1, T2, T3, T4, T5, T6, T7 = [], [], [], [], [], [], []
    for ds, d in D.groupby("dataset", sort=False):
        d = d.reset_index(drop=True); st = stretches(d); y2 = d.y2.to_numpy()
        print(f"\n=== {ds}: {d.subject.nunique()} patients, {len(d) * WIN_S / 3600:.2f} h ===")
        S = strategies(d, st)
        for name, on in S.items():
            c = episode_table(y2, on, st)
            T1.append({"dataset": ds, "strategy": name, "patients": len(c), **boot(c, rng)})
            part = c[["episodes", "timely", "early", "after_onset", "missed"]].sum()
            T2.append({"dataset": ds, "strategy": name, **{f"{k}_%": 100 * part[k] / max(part.episodes, 1) for k in ["timely", "early", "after_onset", "missed"]}})
            if name in (PRIMARY, "Freeze index, floor 0.70", "No hysteresis, floor 0.70"):
                for k, v in c.iterrows():
                    T7.append({"dataset": ds, "strategy": name, **v.to_dict()})
        on = S[PRIMARY]
        for pre in (4, 8, 12):
            for post in (2, 6, 10):
                p = pooled(episode_table(y2, on, st, pre, post))
                T3.append({"dataset": ds, "zone_pre_s": pre * WIN_S, "zone_post_s": post * WIN_S, "false_alarms_per_hour": p["false_alarms_per_hour"]})
        for ml in (0, 2, 4):
            for name in (PRIMARY, "Freeze index, floor 0.70", "No hysteresis, floor 0.70"):
                p = pooled(episode_table(y2, S[name], st, min_lead=ml))
                T4.append({"dataset": ds, "strategy": name, "min_lead_s": ml * WIN_S, "timely_activation_%": p["timely_activation_%"]})
        ps = patient_surrogate(d, st, on, rng); ps.insert(0, "dataset", ds); T5.append(ps)
        print(f"  patient-level surrogate: {int((ps.p_one_sided < 0.05).sum())}/{len(ps)} patients p<0.05")
        sc = d.score.to_numpy(); fi = np.log1p(np.clip(d.fi_score.to_numpy(), 0, None))
        P = pd.DataFrame(curve_points(d, st, sc, "Classifier + hysteresis", True)
                         + curve_points(d, st, sc, "Classifier, single threshold", False)
                         + curve_points(d, st, fi, "Freeze index + hysteresis", True)
                         + curve_points(d, st, fi, "Freeze index, single threshold", False))
        P.insert(0, "dataset", ds); T6.append(P)
        print("  curves done")
    R1 = pd.DataFrame(T1); R1.round(3).to_csv(f"{OUT}/A1_strategies_CI.csv", index=False)
    pd.DataFrame(T2).round(2).to_csv(f"{OUT}/A2_episode_partition.csv", index=False)
    pd.DataFrame(T3).round(2).to_csv(f"{OUT}/A3_tolerance_zone.csv", index=False)
    pd.DataFrame(T4).round(2).to_csv(f"{OUT}/A4_min_lead.csv", index=False)
    pd.concat(T5).round(3).to_csv(f"{OUT}/A5_patient_surrogate.csv", index=False)
    C = pd.concat(T6); C.round(3).to_csv(f"{OUT}/A6_tradeoff_points.csv", index=False)
    F = []
    for ds, g in C.groupby("dataset", sort=False):
        a = frontier(g, "false_alarms_per_hour", FA_TARGETS); a.insert(0, "dataset", ds); F.append(a)
        b = frontier(g, "time_cue_on_%", ON_TARGETS); b.insert(0, "dataset", ds); F.append(b)
    pd.concat(F).round(2).to_csv(f"{OUT}/A7_matched_frontier.csv", index=False)
    pd.DataFrame(T7).round(4).to_csv(f"{OUT}/A8_per_patient.csv", index=False)
    show = ["dataset", "strategy", "coverage_%", "timely_activation_%", "false_alarms_per_hour", "time_cue_on_%"]
    print("\n", R1[show].round(1).to_string(index=False))
    print(f"\nSaved A1-A8 in {OUT}")


if __name__ == "__main__":
    main()
