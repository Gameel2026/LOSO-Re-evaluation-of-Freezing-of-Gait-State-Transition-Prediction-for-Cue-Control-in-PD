import os
import sys
import pickle
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx2_external as ext
import mx2_controller as mx2
import mx_new_controllers as m1

SEEDS = [1, 2, 3, 4]
RUN_COHORTS = list(ext.COHORTS)
OUT = ext.OUT
KEYS = mx2.KEYS


def run_seed_ckpt(ds, X, meta, seed):
    ck = os.path.join(OUT, f"ckpt_{ds.split()[0]}_seed{seed}"); os.makedirs(ck, exist_ok=True)
    cfg = mx2.dataclasses.replace(mx2.fi.cfg_(), seed=seed)
    y2 = meta["y2"].to_numpy(); y4 = mx2.relabel_four_class(meta, m1.PRE_W, m1.POST_W)
    y = (y4 != 0).astype(int); groups = meta["subject"].to_numpy(); n = len(y)
    on_p, on_m, ch = np.zeros(n, bool), np.zeros(n, bool), []
    for s in np.unique(groups):
        f = os.path.join(ck, f"fold_{s}.pkl")
        tr_idx = np.flatnonzero(groups != s); te_idx = np.flatnonzero(groups == s)
        if os.path.exists(f):
            a, b, row = pickle.load(open(f, "rb"))
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
            row = {"dataset": ds, "seed": seed, "subject": s, "primary_on": thp[0], "primary_off": thp[1],
                   "M_on": thm[0], "M_off": thm[1], "M_hold_s": k * mx2.WIN_S, "M_mode": mode}
            pickle.dump((a, b, row), open(f, "wb"))
            print(f"  [{ds} seed {seed}] subject {s}: {mode}, hold {k * mx2.WIN_S:g} s, {mx2.time.time() - t0:.0f} s", flush=True)
        on_p[te_idx], on_m[te_idx] = a, b
        ch.append(row)
    return on_p, on_m, pd.DataFrame(ch)


def seed_file(name, seed):
    return os.path.join(OUT, f"EVS_counts_{name.split()[0]}_seed{seed}.pkl")


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    for seed in SEEDS:
        for name in RUN_COHORTS:
            if os.path.exists(seed_file(name, seed)):
                continue
            X, meta = ext.cache_for(name, ext.COHORTS[name])
            print(f"\n=== {name}, seed {seed} ===", flush=True)
            on_p, on_m, ch = run_seed_ckpt(name, X, meta, seed)
            ch.to_csv(os.path.join(OUT, f"EVS_choices_{name.split()[0]}_seed{seed}.csv"), index=False)
            ev = m1.Ev(meta, meta["y2"].to_numpy(), np.arange(len(meta)))
            pickle.dump((ev.counts(on_m), ev.counts(on_p)), open(seed_file(name, seed), "wb"))
    rng = np.random.default_rng(42)
    T1, T3 = [], []
    for seed in [42] + SEEDS:
        done = {}
        for name in ext.COHORTS:
            f = ext.counts_file(name) if seed == 42 else seed_file(name, seed)
            if os.path.exists(f):
                v = pickle.load(open(f, "rb")); done[name] = (v[0], v[1])
        for name, (cm, cp) in done.items():
            for lab, c in [("Primary rule", cp), ("M", cm)]:
                T1.append({"seed": seed, "cohort": name, "controller": lab, **m1.pooled(c)})
            d = m1.boot_diff(cm, cp, rng, KEYS)
            for k in KEYS:
                T3.append({"seed": seed, "cohort": name, "metric": k, "difference": d[k][0], "CI_low": d[k][1], "CI_high": d[k][2]})
        if len(done) == len(ext.COHORTS):
            p = mx2.boot_pooled(list(done.values()), rng)
            for k in KEYS:
                T3.append({"seed": seed, "cohort": "Pooled (external cohorts)", "metric": k, "difference": p[k][0], "CI_low": p[k][1], "CI_high": p[k][2]})
    A = pd.DataFrame(T1); D = pd.DataFrame(T3)
    A.round(3).to_csv(os.path.join(OUT, "EVS_T1_all_seeds.csv"), index=False)
    D.round(3).to_csv(os.path.join(OUT, "EVS_T3_paired_differences_all_seeds.csv"), index=False)
    R = A.groupby(["cohort", "controller"])[KEYS].agg(["min", "max"]).round(2)
    R.to_csv(os.path.join(OUT, "EVS_T6_seed_range.csv"))
    print("\nRange across seeds\n", R.to_string())
    fa = D[(D.metric == "false_alarms_per_hour")].round(2)
    print("\nFalse-alarm difference (M minus primary rule) by seed\n", fa.to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
