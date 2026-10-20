import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx2_controller as mx2
import mx_new_controllers as m1
import fog_rerun_causal as rc

KAGGLE = os.path.join("kaggle_fog", "daphnet_format")
COHORTS = {"tdcsfog (lab)": os.path.join(KAGGLE, "tdcsfog_back"), "defog (home)": os.path.join(KAGGLE, "defog_back")}
RUN_COHORTS = list(COHORTS)
OUT = "results_mx2_external"
MX2_DIR = "results_mx2"
SEED = 42
KEYS = mx2.KEYS


def cache_for(name, folder):
    sub = os.path.join(OUT, "cache_" + name.split()[0])
    os.makedirs(sub, exist_ok=True)
    rc.OUT = sub
    rc.SENSORS = {"Trunk": ("tr",)}
    rc.SENSOR_ROOTS = {"Trunk": folder}
    path = os.path.join(sub, "cache_Trunk.pkl")
    if not os.path.exists(path):
        print(f"  building features for {name} from {folder} ...", flush=True)
    _, X, meta, _ = rc.data_for(None, "Trunk")
    return X.reset_index(drop=True), meta.reset_index(drop=True)


def counts_file(name):
    return os.path.join(OUT, f"EV_counts_{name.split()[0]}.pkl")


def describe(name, meta):
    y2 = meta["y2"].to_numpy(); n_ep = 0; fog_subj = set()
    for _, g in meta.groupby(["subject", "run", "seg"], sort=True):
        e = len(m1.runs(y2[g.index.to_numpy()] == 2)); n_ep += e
        if e:
            fog_subj.add(g["subject"].iloc[0])
    print(f"  data check {name}: {meta.subject.nunique()} patients ({len(fog_subj)} with FoG), "
          f"{len(meta) * mx2.WIN_S / 3600:.2f} h of windows, {n_ep} episodes at window level, "
          f"FoG windows {100 * (y2 == 2).mean():.1f}%", flush=True)


def run_seed_ckpt(ds, X, meta, seed):
    ck = os.path.join(OUT, "ckpt_" + ds.split()[0]); os.makedirs(ck, exist_ok=True)
    cfg = mx2.dataclasses.replace(mx2.fi.cfg_(), seed=seed)
    y2 = meta["y2"].to_numpy(); y4 = mx2.relabel_four_class(meta, m1.PRE_W, m1.POST_W)
    y = (y4 != 0).astype(int); groups = meta["subject"].to_numpy(); n = len(y)
    on_p, on_m, sp, sm, ch = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n), np.zeros(n), []
    for s in np.unique(groups):
        f = os.path.join(ck, f"fold_{s}.pkl")
        tr_idx = np.flatnonzero(groups != s); te_idx = np.flatnonzero(groups == s)
        if os.path.exists(f):
            a, b, c, d, row = pickle.load(open(f, "rb"))
        else:
            t0 = mx2.time.time()
            ev_tr, ev_te = m1.Ev(meta, y2, tr_idx), m1.Ev(meta, y2, te_idx)
            feats = mx2.fi.rank_mi(X.iloc[tr_idx], y[tr_idx], cfg.seed).index[:mx2.K_FEATS].tolist()
            p_oof, p_te = mx2.scores(X, y, tr_idx, te_idx, groups, feats, cfg, mx2.TREES_PRIMARY)
            m_oof, m_te = mx2.scores(X, y, tr_idx, te_idx, groups, feats, cfg, mx2.TREES_M)
            thp = m1.select_primary(ev_tr, p_oof[tr_idx], y[tr_idx])
            a = ev_te.hyst(p_te, *thp)
            thm, k, mode, c_ref, t_ref = mx2.select_m(ev_tr, m_oof[tr_idx], y[tr_idx])
            b = mx2.hold_k(ev_te, ev_te.hyst(m_te, *thm), k)
            c, d = p_te, m_te
            row = {"dataset": ds, "seed": seed, "subject": s, "primary_on": thp[0], "primary_off": thp[1],
                   "M_on": thm[0], "M_off": thm[1], "M_hold_s": k * mx2.WIN_S, "M_mode": mode,
                   "inner_ref_coverage": c_ref, "inner_ref_time_on": t_ref}
            pickle.dump((a, b, c, d, row), open(f, "wb"))
            print(f"  [{ds} seed {seed}] subject {s}: {mode}, hold {k * mx2.WIN_S:g} s, {mx2.time.time() - t0:.0f} s", flush=True)
        on_p[te_idx], on_m[te_idx], sp[te_idx], sm[te_idx] = a, b, c, d
        ch.append(row)
    return on_p, on_m, sp, sm, pd.DataFrame(ch)


def run_cohort(name, folder):
    t0 = mx2.time.time()
    X, meta = cache_for(name, folder)
    print(f"  features ready in {mx2.time.time() - t0:.0f} s", flush=True)
    describe(name, meta)
    ev = m1.Ev(meta, meta["y2"].to_numpy(), np.arange(len(meta)))
    print(f"\n=== {name}: {meta.subject.nunique()} patients, {len(meta) * mx2.WIN_S / 3600:.2f} h ===", flush=True)
    on_p, on_m, sp, sm, ch = run_seed_ckpt(name, X, meta, SEED)
    W = meta[["subject", "run", "seg", "pos", "y2"]].copy()
    W["score | primary"] = sp.round(5); W["score | M"] = sm.round(5)
    W["on | Primary"] = on_p.astype(np.int8); W["on | M"] = on_m.astype(np.int8)
    W.to_csv(os.path.join(OUT, f"EV_per_window_{name.split()[0]}_seed{SEED}.csv.gz"), index=False, compression="gzip")
    ch.to_csv(os.path.join(OUT, f"EV_choices_{name.split()[0]}.csv"), index=False)
    cp, cm = ev.counts(on_p), ev.counts(on_m)
    sur = m1.surrogate(ev, on_m, np.random.default_rng(SEED))
    pickle.dump((cm, cp, sur), open(counts_file(name), "wb"))
    return cm, cp


def mx2_counts():
    out = {}
    for ds in m1.DATASETS:
        f = os.path.join(MX2_DIR, f"MX2_per_window_{ds.replace(' ', '_')}_seed{SEED}.csv.gz")
        if not os.path.exists(f):
            return None
        W = pd.read_csv(f)
        ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
        out[ds] = (ev.counts(W["on | M"].to_numpy().astype(bool)), ev.counts(W["on | Primary"].to_numpy().astype(bool)))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    for name in RUN_COHORTS:
        if os.path.exists(counts_file(name)):
            print(f"{name}: results already saved - skipped (delete {counts_file(name)} to rerun)")
            continue
        folder = COHORTS[name]
        if not glob.glob(os.path.join(folder, "S*R*.txt")):
            print(f"{name}: no converted files in {folder} - run kaggle_convert.py first"); continue
        run_cohort(name, folder)
    done = {n: pickle.load(open(counts_file(n), "rb")) for n in COHORTS if os.path.exists(counts_file(n))}
    if not done:
        return
    rng = np.random.default_rng(SEED)
    T_ci, T_diff, T_h, T_sur = [], [], [], []
    for name, (cm, cp, sur) in done.items():
        for lab, c in [("Primary rule", cp), ("M", cm)]:
            ci = m1.boot_ci(c, rng, KEYS)
            T_ci.append({"cohort": name, "controller": lab, **{k: ci[k][0] for k in KEYS},
                         **{f"{k}_CI_low": ci[k][1] for k in KEYS}, **{f"{k}_CI_high": ci[k][2] for k in KEYS}})
        d = m1.boot_diff(cm, cp, rng, KEYS)
        for k in KEYS:
            T_diff.append({"cohort": name, "comparison": "M - Primary rule", "metric": k, "difference": d[k][0], "CI_low": d[k][1], "CI_high": d[k][2]})
        for hyp, k, margin, lower in [("EV2 coverage non-inferiority (-10)", "coverage_%", mx2.MARGIN_COV, True),
                                      ("EV3 cue-on time non-inferiority (+5)", "time_cue_on_%", mx2.MARGIN_ON, False),
                                      ("EV4 lead >= 1 s non-inferiority (-10)", "timely_lead1_%", mx2.MARGIN_LEAD, True)]:
            T_h.append({"cohort": name, "hypothesis": hyp, "estimate": d[k][0], "CI_low": d[k][1], "CI_high": d[k][2],
                        "result": mx2.noninf(d[k][1], d[k][2], margin, lower)})
        for r in sur:
            T_sur.append({"cohort": name, "controller": "M", **r})
    if len(done) == len(COHORTS):
        p = mx2.boot_pooled([(v[0], v[1]) for v in done.values()], rng)
        for k in KEYS:
            T_diff.append({"cohort": "Pooled (external cohorts)", "comparison": "M - Primary rule", "metric": k,
                           "difference": p[k][0], "CI_low": p[k][1], "CI_high": p[k][2]})
        f = p["false_alarms_per_hour"]
        T_h.insert(0, {"cohort": "Pooled (external cohorts)", "hypothesis": "EV1 fewer false alarms per hour (primary)",
                       "estimate": f[0], "CI_low": f[1], "CI_high": f[2], "result": "supported" if f[2] < 0 else "not supported"})
        m2 = mx2_counts()
        if m2 is not None:
            q = mx2.boot_pooled([(v[0], v[1]) for v in done.values()] + list(m2.values()), rng)
            for k in KEYS:
                T_diff.append({"cohort": "Pooled (five cohorts, secondary)", "comparison": "M - Primary rule", "metric": k,
                               "difference": q[k][0], "CI_low": q[k][1], "CI_high": q[k][2]})
        else:
            print(f"MX-2 per-window files not found in {MX2_DIR}; five-cohort pooled estimate skipped")
    pd.DataFrame(T_ci).round(3).to_csv(os.path.join(OUT, "EV_T2_controllers_CI.csv"), index=False)
    pd.DataFrame(T_diff).round(3).to_csv(os.path.join(OUT, "EV_T3_paired_differences.csv"), index=False)
    H = pd.DataFrame(T_h); H.round(3).to_csv(os.path.join(OUT, "EV_T4_hypotheses.csv"), index=False)
    pd.DataFrame(T_sur).round(3).to_csv(os.path.join(OUT, "EV_T5_surrogate.csv"), index=False)
    print("\n", pd.DataFrame(T_ci)[["cohort", "controller"] + KEYS].round(1).to_string(index=False))
    print("\nHypotheses\n", H.round(2).to_string(index=False))
    D = pd.DataFrame(T_diff)
    print("\nPooled false alarms\n", D[D.cohort.str.startswith("Pooled") & (D.metric == "false_alarms_per_hour")].round(2).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()