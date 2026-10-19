import os
import sys
import time
import pickle
import dataclasses
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

import fog_improve as fi
from fog_pipeline import relabel_four_class
from fog_fix3 import Model

OUT = "results_mx2"
DATASETS = m1.DATASETS
REFERENCE = m1.REFERENCE
SEEDS = [42, 1, 2, 3, 4]
PRIMARY_SEED = 42
TREES_PRIMARY = (50, 100)
TREES_M = (250, 500)
HOLDS = [0, 2, 4, 6]
SPEC_MIN = 0.70
WIN_S = m1.WIN_S
K_FEATS = 23
N_BOOT = 5000
MARGIN_COV, MARGIN_ON, MARGIN_LEAD = 10.0, 5.0, 10.0
GRID = m1.GRID
KEYS = ["coverage_%", "timely_%", "timely_lead1_%", "cue_at_onset_%", "false_alarms_per_hour", "time_cue_on_%"]
COUNT_COLS = ["episodes", "detected", "timely", "timely_lead1", "cue_at_onset", "false_alarms", "windows", "on_windows"]


def hold_k(ev, h, k):
    out = h.copy()
    for j in range(1, k + 1):
        out |= np.r_[np.zeros(j, bool), h[:-j]] & (np.r_[np.full(j, -1), ev.sid[:-j]] == ev.sid)
    return out


def scores(X, y, tr_idx, te_idx, groups, feats, cfg, trees):
    oof = np.zeros(len(y))
    for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
        m = Model("RF", cfg, n_trees=trees[0]).fit(X.iloc[tr_idx[a]][feats].to_numpy(), y[tr_idx[a]])
        oof[tr_idx[b]] = m1.proba1(m, X.iloc[tr_idx[b]][feats].to_numpy())
    m = Model("RF", cfg, n_trees=trees[1]).fit(X.iloc[tr_idx][feats].to_numpy(), y[tr_idx])
    return oof, m1.proba1(m, X.iloc[te_idx][feats].to_numpy())


def select_m(ev, oof, y):
    th_ref = m1.select_primary(ev, oof, y)
    ref = m1.pooled(ev.counts(ev.hyst(oof, *th_ref)))
    c_ref, t_ref = ref["coverage_%"], ref["time_cue_on_%"]
    rows = []
    for th in GRID:
        h = ev.hyst(oof, *th)
        for k in HOLDS:
            p = m1.pooled(ev.counts(hold_k(ev, h, k))); p["th"] = th; p["hold"] = k
            rows.append(p)
    feas = [r for r in rows if r["coverage_%"] >= c_ref - 1e-9 and r["time_cue_on_%"] <= t_ref + 1e-9 and r["spec_outside_zone"] >= SPEC_MIN]
    if feas:
        r = min(feas, key=lambda r: (r["false_alarms_per_hour"], -r["timely_lead1_%"], r["hold"]))
        return r["th"], r["hold"], "dominant", c_ref, t_ref
    burden = [r for r in rows if r["time_cue_on_%"] <= t_ref + 1e-9]
    if burden:
        r = max(burden, key=lambda r: (r["coverage_%"], -r["false_alarms_per_hour"], -r["hold"]))
        return r["th"], r["hold"], "burden-only", c_ref, t_ref
    return th_ref, 0, "primary rule", c_ref, t_ref


def run_seed(ds, X, meta, seed):
    cfg = dataclasses.replace(fi.cfg_(), seed=seed)
    y2 = meta["y2"].to_numpy(); y4 = relabel_four_class(meta, m1.PRE_W, m1.POST_W)
    y = (y4 != 0).astype(int); groups = meta["subject"].to_numpy(); n = len(y)
    on_p, on_m = np.zeros(n, bool), np.zeros(n, bool); sp, sm = np.zeros(n), np.zeros(n); ch = []
    for s in np.unique(groups):
        t0 = time.time()
        tr_idx = np.flatnonzero(groups != s); te_idx = np.flatnonzero(groups == s)
        ev_tr, ev_te = m1.Ev(meta, y2, tr_idx), m1.Ev(meta, y2, te_idx)
        feats = fi.rank_mi(X.iloc[tr_idx], y[tr_idx], cfg.seed).index[:K_FEATS].tolist()
        p_oof, p_te = scores(X, y, tr_idx, te_idx, groups, feats, cfg, TREES_PRIMARY)
        m_oof, m_te = scores(X, y, tr_idx, te_idx, groups, feats, cfg, TREES_M)
        thp = m1.select_primary(ev_tr, p_oof[tr_idx], y[tr_idx])
        on_p[te_idx] = ev_te.hyst(p_te, *thp)
        thm, k, mode, c_ref, t_ref = select_m(ev_tr, m_oof[tr_idx], y[tr_idx])
        on_m[te_idx] = hold_k(ev_te, ev_te.hyst(m_te, *thm), k)
        sp[te_idx], sm[te_idx] = p_te, m_te
        ch.append({"dataset": ds, "seed": seed, "subject": s, "primary_on": thp[0], "primary_off": thp[1],
                   "M_on": thm[0], "M_off": thm[1], "M_hold_s": k * WIN_S, "M_mode": mode,
                   "inner_ref_coverage": c_ref, "inner_ref_time_on": t_ref})
        print(f"  [{ds} seed {seed}] subject {s}: {mode}, hold {k * WIN_S:g} s, {time.time() - t0:.0f} s", flush=True)
    return on_p, on_m, sp, sm, pd.DataFrame(ch)


def boot_pooled(pairs, rng):
    tot_a = {k: 0 for k in COUNT_COLS}; tot_b = {k: 0 for k in COUNT_COLS}; pa = {k: 0.0 for k in COUNT_COLS}; pb = {k: 0.0 for k in COUNT_COLS}
    for ca, cb in pairs:
        A = ca.set_index("subject"); B = cb.set_index("subject").reindex(A.index)
        idx = rng.integers(0, len(A), (N_BOOT, len(A)))
        for k in COUNT_COLS:
            tot_a[k] = tot_a[k] + A[k].to_numpy(float)[idx].sum(1); tot_b[k] = tot_b[k] + B[k].to_numpy(float)[idx].sum(1)
            pa[k] += A[k].sum(); pb[k] += B[k].sum()
    def met(d):
        e = np.maximum(d["episodes"], 1); h = d["windows"] * WIN_S / 3600
        return {"coverage_%": 100 * d["detected"] / e, "timely_%": 100 * d["timely"] / e, "timely_lead1_%": 100 * d["timely_lead1"] / e,
                "cue_at_onset_%": 100 * d["cue_at_onset"] / e, "false_alarms_per_hour": d["false_alarms"] / h,
                "time_cue_on_%": 100 * d["on_windows"] / d["windows"]}
    ma, mb, ea, eb = met(tot_a), met(tot_b), met(pa), met(pb)
    return {k: (ea[k] - eb[k], np.percentile(ma[k] - mb[k], 2.5), np.percentile(ma[k] - mb[k], 97.5)) for k in KEYS}


def noninf(lo, hi, margin, lower=True):
    ok = lo > -margin if lower else hi < margin
    if ok:
        return "supported"
    return "inconclusive" if (hi - lo) / 2 > margin else "not supported"


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(PRIMARY_SEED)
    ref = pd.read_csv(REFERENCE) if os.path.exists(REFERENCE) else None
    T_all, T_ci, T_diff, T_h, T_sur, T_ch = [], [], [], [], [], []
    primary_counts = {}
    for ds, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"{ds}: {path} not found - skipped"); continue
        _, X, meta, _ = pickle.load(open(path, "rb"))
        X = X.reset_index(drop=True); meta = meta.reset_index(drop=True)
        ev = m1.Ev(meta, meta["y2"].to_numpy(), np.arange(len(meta)))
        print(f"\n=== {ds}: {meta.subject.nunique()} patients, {len(meta) * WIN_S / 3600:.2f} h ===")
        for seed in SEEDS:
            on_p, on_m, sp, sm, ch = run_seed(ds, X, meta, seed); T_ch.append(ch)
            W = meta[["subject", "run", "seg", "pos", "y2"]].copy()
            W["score | primary"] = sp.round(5); W["score | M"] = sm.round(5)
            W["on | Primary"] = on_p.astype(np.int8); W["on | M"] = on_m.astype(np.int8)
            W.to_csv(f"{OUT}/MX2_per_window_{ds.replace(' ', '_')}_seed{seed}.csv.gz", index=False, compression="gzip")
            for name, v in [("Primary", on_p), ("M", on_m)]:
                T_all.append({"dataset": ds, "seed": seed, "controller": name, **m1.pooled(ev.counts(v))})
            pd.DataFrame(T_all).round(3).to_csv(f"{OUT}/MX2_T1_all_seeds.csv", index=False)
            pd.concat(T_ch).to_csv(f"{OUT}/MX2_choices.csv", index=False)
            if seed != PRIMARY_SEED:
                continue
            if ref is not None:
                r = ref[ref.dataset == ds].reset_index(drop=True)
                print(f"  check: primary score equals rerun score: {np.allclose(sp.round(5), r['score'].to_numpy(), atol=1e-4)}; "
                      f"primary on-vector equals frozen primary: {(on_p == r['on | Controller, floor 0.70'].astype(bool)).mean():.4f}")
            cp, cm = ev.counts(on_p), ev.counts(on_m); primary_counts[ds] = (cm, cp)
            for name, c in [("Primary", cp), ("M", cm)]:
                ci = m1.boot_ci(c, rng, KEYS)
                T_ci.append({"dataset": ds, "controller": name, **{k: ci[k][0] for k in KEYS},
                             **{f"{k}_CI_low": ci[k][1] for k in KEYS}, **{f"{k}_CI_high": ci[k][2] for k in KEYS}})
            d = m1.boot_diff(cm, cp, rng, KEYS)
            for k in KEYS:
                T_diff.append({"dataset": ds, "comparison": "M - Primary", "metric": k, "difference": d[k][0], "CI_low": d[k][1], "CI_high": d[k][2]})
            T_h.append({"dataset": ds, "hypothesis": "H2 coverage non-inferiority (-10)", "estimate": d["coverage_%"][0],
                        "CI_low": d["coverage_%"][1], "CI_high": d["coverage_%"][2], "result": noninf(d["coverage_%"][1], d["coverage_%"][2], MARGIN_COV)})
            T_h.append({"dataset": ds, "hypothesis": "H3 cue-on time non-inferiority (+5)", "estimate": d["time_cue_on_%"][0],
                        "CI_low": d["time_cue_on_%"][1], "CI_high": d["time_cue_on_%"][2], "result": noninf(d["time_cue_on_%"][1], d["time_cue_on_%"][2], MARGIN_ON, lower=False)})
            T_h.append({"dataset": ds, "hypothesis": "H4 lead >= 1 s non-inferiority (-10)", "estimate": d["timely_lead1_%"][0],
                        "CI_low": d["timely_lead1_%"][1], "CI_high": d["timely_lead1_%"][2], "result": noninf(d["timely_lead1_%"][1], d["timely_lead1_%"][2], MARGIN_LEAD)})
            for r in m1.surrogate(ev, on_m, rng):
                T_sur.append({"dataset": ds, "controller": "M", **r})
            pd.DataFrame(T_ci).round(3).to_csv(f"{OUT}/MX2_T2_controllers_CI.csv", index=False)
            pd.DataFrame(T_diff).round(3).to_csv(f"{OUT}/MX2_T3_paired_differences.csv", index=False)
            pd.DataFrame(T_sur).round(3).to_csv(f"{OUT}/MX2_T5_surrogate.csv", index=False)
            print(pd.DataFrame([x for x in T_ci if x["dataset"] == ds])[["controller"] + KEYS].round(1).to_string(index=False))
    if len(primary_counts) == len(DATASETS):
        pdiff = boot_pooled(list(primary_counts.values()), rng)
        for k in KEYS:
            T_diff.append({"dataset": "Pooled (three cohorts)", "comparison": "M - Primary", "metric": k,
                           "difference": pdiff[k][0], "CI_low": pdiff[k][1], "CI_high": pdiff[k][2]})
        f = pdiff["false_alarms_per_hour"]
        T_h.insert(0, {"dataset": "Pooled (three cohorts)", "hypothesis": "H1 fewer false alarms per hour", "estimate": f[0],
                       "CI_low": f[1], "CI_high": f[2], "result": "supported" if f[2] < 0 else "not supported"})
    A = pd.DataFrame(T_all)
    for ds in A.dataset.unique():
        g = A[A.dataset == ds]
        if g.seed.nunique() < len(SEEDS):
            continue
        rp = g[g.controller == "Primary"].false_alarms_per_hour; rm = g[g.controller == "M"].false_alarms_per_hour
        T_h.append({"dataset": ds, "hypothesis": "H5 narrower false-alarm range across seeds", "estimate": (rm.max() - rm.min()) - (rp.max() - rp.min()),
                    "CI_low": np.nan, "CI_high": np.nan, "result": "supported" if (rm.max() - rm.min()) < (rp.max() - rp.min()) else "not supported",
                    "primary_range": f"{rp.min():.1f}-{rp.max():.1f}", "M_range": f"{rm.min():.1f}-{rm.max():.1f}"})
    pd.DataFrame(T_diff).round(3).to_csv(f"{OUT}/MX2_T3_paired_differences.csv", index=False)
    H = pd.DataFrame(T_h); H.round(3).to_csv(f"{OUT}/MX2_T4_hypotheses.csv", index=False)
    rng_tab = A.groupby(["dataset", "controller"])[KEYS].agg(["min", "max"]).round(2); rng_tab.to_csv(f"{OUT}/MX2_T6_seed_range.csv")
    C = pd.concat(T_ch); C.to_csv(f"{OUT}/MX2_choices.csv", index=False)
    C.groupby(["dataset", "M_mode", "M_hold_s"]).size().rename("folds").reset_index().to_csv(f"{OUT}/MX2_T7_selection_modes.csv", index=False)
    print("\nHypotheses\n", H.round(2).to_string(index=False))
    print("\nRange across seeds\n", rng_tab.to_string())
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()