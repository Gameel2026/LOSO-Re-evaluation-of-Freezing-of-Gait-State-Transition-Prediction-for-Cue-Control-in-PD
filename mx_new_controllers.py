import os
import sys
import time
import pickle
import dataclasses
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
BASE = next((c for c in CANDIDATES if os.path.isdir(os.path.join(c, "results_causal"))), HERE)
os.chdir(BASE)
sys.path.insert(0, BASE)

import fog_improve as fi
from fog_pipeline import relabel_four_class
from fog_fix3 import Model

OUT = "results_mx1"
DATASETS = {"DAPHNET": "results_causal/cache_Shank.pkl",
            "Multimodal FoG": "results_li_causal/cache_Shank.pkl",
            "FoG-STAR": "results_fogstar_causal/cache_Shank.pkl"}
REFERENCE = "results_methodsx/MX_per_window.csv.gz"
SEEDS = [42, 1, 2, 3, 4]
PRIMARY_SEED = 42
BUDGETS = [60, 30, 120]
PRIMARY_BUDGET = 60
SPEC_MIN = 0.70
PRE_W, POST_W, HOLD = 8, 6, 6
WIN_S = 0.5
K_FEATS = 23
N_BOOT = 5000
N_SHIFT = 200
GRID = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
NEW = ["U*", "U* (no hold)", "D", "A"]


def runs(m):
    e = np.flatnonzero(np.diff(np.r_[0, m.astype(np.int8), 0]))
    return list(zip(e[::2], e[1::2]))


class Ev:
    def __init__(self, meta, y2, idx):
        m = meta.iloc[idx]
        self.n = len(idx)
        key = m["subject"].astype(str) + "_" + m["run"].astype(str) + "_" + m["seg"].astype(str)
        self.sid = pd.factorize(key)[0]
        self.first = np.r_[True, self.sid[1:] != self.sid[:-1]]
        self.subj = m["subject"].to_numpy()
        self.fog = y2[idx] == 2
        zone = np.zeros(self.n, bool); eps = []
        starts = np.flatnonzero(self.first); ends = np.r_[starts[1:], self.n]
        for a, b in zip(starts, ends):
            prev = None
            for s, e in runs(self.fog[a:b]):
                s, e = s + a, e + a
                zone[max(a, s - PRE_W):min(b, e + POST_W)] = True
                eps.append((s, e, a, prev[0] if prev else -1, prev[1] if prev else -1))
                prev = (s, e)
        self.zone, self.outside = zone, ~zone
        E = np.array(eps, dtype=int).reshape(-1, 5)
        self.es, self.ee, self.ea, self.ps, self.pe = E.T
        self.esubj = self.subj[self.es] if len(E) else np.array([], int)
        self.subjects = np.unique(self.subj)
        self.win_per = pd.Series(1, index=self.subj).groupby(level=0).sum().reindex(self.subjects).to_numpy()

    def hyst(self, score, on, off):
        lab = np.where(score >= on, 1, np.where(score < off, 0, -1))
        lab[self.first & (lab == -1)] = 0
        i = np.where(lab >= 0, np.arange(self.n), 0)
        i = np.maximum.accumulate(i)
        return lab[i] == 1

    def hold(self, h):
        out = h.copy()
        for j in range(1, HOLD + 1):
            sh = np.r_[np.zeros(j, bool), h[:-j]] & np.r_[np.full(j, -1), self.sid[:-j]].__eq__(self.sid)
            out |= sh
        return out

    def counts(self, cue):
        prev = np.r_[False, cue[:-1]] & ~self.first
        rs = cue & ~prev
        rid = np.cumsum(rs) * cue
        nr = int(rs.sum())
        zr = np.bincount(rid[cue], weights=self.zone[cue].astype(float), minlength=nr + 1)[1:] if nr else np.zeros(0)
        rsubj = self.subj[np.flatnonzero(rs)]
        fa = pd.Series((zr == 0).astype(int), index=rsubj).groupby(level=0).sum().reindex(self.subjects, fill_value=0).to_numpy()
        c = np.r_[0, np.cumsum(cue)]
        det = (c[self.ee] - c[self.es]) > 0
        rstart = np.maximum.accumulate(np.where(rs, np.arange(self.n), -1))
        has_pre = self.es > self.ea
        pm1 = np.where(has_pre, self.es - 1, 0)
        pre = has_pre & cue[pm1]
        st = rstart[pm1]
        lead = self.es - st
        timely = pre & (lead <= PRE_W)
        lead1 = pre & (lead >= 2) & (lead <= PRE_W)
        carry = pre & (self.ps >= 0) & (st >= self.ps) & (st < self.pe + POST_W)
        onset = lead1 | carry
        def per(v):
            return pd.Series(v.astype(int), index=self.esubj).groupby(level=0).sum().reindex(self.subjects, fill_value=0).to_numpy()
        onw = pd.Series(cue.astype(int), index=self.subj).groupby(level=0).sum().reindex(self.subjects).to_numpy()
        outw = pd.Series(self.outside.astype(int), index=self.subj).groupby(level=0).sum().reindex(self.subjects).to_numpy()
        offout = pd.Series((self.outside & ~cue).astype(int), index=self.subj).groupby(level=0).sum().reindex(self.subjects).to_numpy()
        return pd.DataFrame({"subject": self.subjects, "episodes": per(np.ones(len(self.es), bool)), "detected": per(det),
                             "timely": per(timely), "timely_lead1": per(lead1), "cue_at_onset": per(onset),
                             "false_alarms": fa, "windows": self.win_per, "on_windows": onw,
                             "outside_windows": outw, "outside_off": offout})


def pooled(c):
    e = max(c["episodes"].sum(), 1); h = c["windows"].sum() * WIN_S / 3600
    return {"episodes": int(c["episodes"].sum()), "coverage_%": 100 * c["detected"].sum() / e,
            "timely_%": 100 * c["timely"].sum() / e, "timely_lead1_%": 100 * c["timely_lead1"].sum() / e,
            "cue_at_onset_%": 100 * c["cue_at_onset"].sum() / e, "false_alarms_per_hour": c["false_alarms"].sum() / h,
            "time_cue_on_%": 100 * c["on_windows"].sum() / c["windows"].sum(),
            "spec_outside_zone": c["outside_off"].sum() / max(c["outside_windows"].sum(), 1)}


def window_rates(y, p):
    t = y.astype(bool); tp = (p & t).sum(); fp = (p & ~t).sum(); fn = (~p & t).sum()
    sens = tp / max(t.sum(), 1); spec = (~p & ~t).sum() / max((~t).sum(), 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return sens, spec, f1


def select_primary(ev, score, y):
    best, bv = None, (-1, -1.0)
    for th in GRID:
        sens, spec, f1 = window_rates(y, ev.hyst(score, *th))
        v = (1, f1) if spec >= SPEC_MIN else (0, 0.5 * (sens + spec))
        if v > bv:
            best, bv = th, v
    return best


def candidate_table(ev, score, hold):
    rows = []
    for th in GRID:
        cue = ev.hyst(score, *th)
        if hold:
            cue = ev.hold(cue)
        p = pooled(ev.counts(cue)); p["th"] = th
        rows.append(p)
    return rows


def select_event(rows, objective, budget):
    feas = [r for r in rows if r["false_alarms_per_hour"] <= budget and r["spec_outside_zone"] >= SPEC_MIN]
    if feas:
        return max(feas, key=lambda r: (r[objective], -r["false_alarms_per_hour"], -r["time_cue_on_%"]))["th"], True
    spec_ok = [r for r in rows if r["spec_outside_zone"] >= SPEC_MIN] or rows
    return min(spec_ok, key=lambda r: (r["false_alarms_per_hour"], -r[objective]))["th"], False


def proba1(m, X):
    P, cl = m.predict_proba(X)
    cl = list(cl)
    return P[:, cl.index(1)] if 1 in cl else np.zeros(len(X))


def fit_scores(X, y, rows_fit, tr_idx, te_idx, groups, cfg):
    sub = tr_idx[rows_fit[tr_idx]]
    feats = fi.rank_mi(X.iloc[sub], y[sub], cfg.seed).index[:K_FEATS].tolist()
    oof = np.zeros(len(y))
    for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
        fa = tr_idx[a][rows_fit[tr_idx[a]]]
        m = Model("RF", cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[fa][feats].to_numpy(), y[fa])
        oof[tr_idx[b]] = proba1(m, X.iloc[tr_idx[b]][feats].to_numpy())
    m = Model("RF", cfg).fit(X.iloc[sub][feats].to_numpy(), y[sub])
    test = proba1(m, X.iloc[te_idx][feats].to_numpy())
    return oof, test


def run_seed(ds, X, meta, seed):
    cfg = dataclasses.replace(fi.cfg_(), seed=seed)
    y2 = meta["y2"].to_numpy(); y4 = relabel_four_class(meta, PRE_W, POST_W)
    y_union = (y4 != 0).astype(int); y_fog = (y2 == 2).astype(int)
    ant_pos = np.zeros(len(y4), bool)
    for _, g in meta.groupby(["subject", "run", "seg"], sort=True):
        idx = g.index.to_numpy()
        for s, e in runs(y2[idx] == 2):
            for j in range(max(0, s - PRE_W), s - 1):
                if y4[idx[j]] == 1:
                    ant_pos[idx[j]] = True
    y_ant = ant_pos.astype(int); ant_rows = ant_pos | (y4 == 0)
    y_comb = (y_fog.astype(bool) | ant_pos).astype(int)
    allrows = np.ones(len(y4), bool)
    groups = meta["subject"].to_numpy(); n = len(y4)
    S = {k: np.zeros(n) for k in ["union", "fog", "ant", "comb"]}
    ON = {"Primary": np.zeros(n, bool)}
    for c in NEW:
        for b in BUDGETS:
            ON[f"{c} | {b}"] = np.zeros(n, bool)
    choices = []
    for s in np.unique(groups):
        t0 = time.time()
        tr_idx = np.flatnonzero(groups != s); te_idx = np.flatnonzero(groups == s)
        ev_tr, ev_te = Ev(meta, y2, tr_idx), Ev(meta, y2, te_idx)
        u_oof, u_te = fit_scores(X, y_union, allrows, tr_idx, te_idx, groups, cfg)
        d_oof, d_te = fit_scores(X, y_fog, allrows, tr_idx, te_idx, groups, cfg)
        a_oof, a_te = fit_scores(X, y_ant, ant_rows, tr_idx, te_idx, groups, cfg)
        lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(np.c_[d_oof[tr_idx], a_oof[tr_idx]], y_comb[tr_idx])
        c_oof = lr.predict_proba(np.c_[d_oof[tr_idx], a_oof[tr_idx]])[:, 1]
        c_te = lr.predict_proba(np.c_[d_te, a_te])[:, 1]
        S["union"][te_idx], S["fog"][te_idx], S["ant"][te_idx], S["comb"][te_idx] = u_te, d_te, a_te, c_te
        th = select_primary(ev_tr, u_oof[tr_idx], y_union[tr_idx])
        ON["Primary"][te_idx] = ev_te.hyst(u_te, *th)
        row = {"dataset": ds, "seed": seed, "subject": s, "Primary": th,
               "combiner_coef_det": lr.coef_[0][0], "combiner_coef_ant": lr.coef_[0][1], "combiner_intercept": lr.intercept_[0]}
        tabs = {"U*": (candidate_table(ev_tr, u_oof[tr_idx], True), "coverage_%", u_te, True),
                "U* (no hold)": (candidate_table(ev_tr, u_oof[tr_idx], False), "coverage_%", u_te, False),
                "D": (candidate_table(ev_tr, d_oof[tr_idx], True), "coverage_%", d_te, True),
                "A": (candidate_table(ev_tr, c_oof, True), "cue_at_onset_%", c_te, True)}
        for c, (rows, obj, sc, hold) in tabs.items():
            for b in BUDGETS:
                thc, feas = select_event(rows, obj, b)
                cue = ev_te.hyst(sc, *thc)
                ON[f"{c} | {b}"][te_idx] = ev_te.hold(cue) if hold else cue
                row[f"{c} | {b}"] = thc; row[f"{c} | {b} | feasible"] = feas
        choices.append(row)
        print(f"  [{ds} seed {seed}] subject {s}: {time.time() - t0:.0f} s", flush=True)
    return S, ON, pd.DataFrame(choices), y4


def boot_diff(ca, cb, rng, keys):
    A = ca.set_index("subject"); B = cb.set_index("subject").reindex(A.index)
    n = len(A); idx = rng.integers(0, n, (N_BOOT, n))
    cols = ["episodes", "detected", "timely", "timely_lead1", "cue_at_onset", "false_alarms", "windows", "on_windows"]
    a = {k: A[k].to_numpy(float)[idx].sum(1) for k in cols}; b = {k: B[k].to_numpy(float)[idx].sum(1) for k in cols}
    def met(d, k):
        e = np.maximum(d["episodes"], 1); h = d["windows"] * WIN_S / 3600
        return {"coverage_%": 100 * d["detected"] / e, "timely_%": 100 * d["timely"] / e,
                "timely_lead1_%": 100 * d["timely_lead1"] / e, "cue_at_onset_%": 100 * d["cue_at_onset"] / e,
                "false_alarms_per_hour": d["false_alarms"] / h, "time_cue_on_%": 100 * d["on_windows"] / d["windows"]}[k]
    pa, pb = pooled(ca), pooled(cb); out = {}
    for k in keys:
        diff = met(a, k) - met(b, k)
        out[k] = (pa[k] - pb[k], np.percentile(diff, 2.5), np.percentile(diff, 97.5))
    return out


def boot_ci(c, rng, keys):
    zero = c.copy()
    for k in ["detected", "timely", "timely_lead1", "cue_at_onset", "false_alarms", "on_windows"]:
        zero[k] = 0
    return boot_diff(c, zero, rng, keys)


def surrogate(ev, cue, rng):
    obs = pooled(ev.counts(cue)); null = []
    starts = np.flatnonzero(ev.first); ends = np.r_[starts[1:], ev.n]
    for _ in range(N_SHIFT):
        sh = cue.copy()
        for a, b in zip(starts, ends):
            if b - a > 1:
                sh[a:b] = np.roll(cue[a:b], rng.integers(1, b - a))
        p = pooled(ev.counts(sh)); null.append([p["timely_lead1_%"], p["cue_at_onset_%"], p["timely_%"]])
    null = np.array(null); rows = []
    for j, k in enumerate(["timely_lead1_%", "cue_at_onset_%", "timely_%"]):
        rows.append({"metric": k, "observed": obs[k], "surrogate_mean": null[:, j].mean(),
                     "surrogate_95th": np.quantile(null[:, j], 0.95), "p_one_sided": (1 + (null[:, j] >= obs[k]).sum()) / (1 + N_SHIFT)})
    return rows


def verdict_h(ds, D):
    v = []
    fa = D[("U*", "Primary")]["false_alarms_per_hour"]; cv = D[("U*", "Primary")]["coverage_%"]
    v.append(("H1 U* vs primary", "supported" if (fa[2] < 0 and cv[1] > -5) else "not supported"))
    cv = D[("D", "U*")]["coverage_%"]; fa = D[("D", "U*")]["false_alarms_per_hour"]
    v.append(("H2 D vs U*", "supported" if (cv[1] > 0 and fa[2] <= 10) else "not supported"))
    t = D[("A", "D")]["timely_lead1_%"]
    if ds == "DAPHNET":
        v.append(("H3 A vs D (lead >= 1 s)", "supported" if t[1] > 0 else "not supported"))
    else:
        r = "supported" if (t[1] >= -5 and t[2] <= 5) else ("not supported" if (t[1] > 5 or t[2] < -5) else "inconclusive")
        v.append(("H4 A vs D (lead >= 1 s), equivalence +/-5", r))
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(PRIMARY_SEED)
    ref = pd.read_csv(REFERENCE) if os.path.exists(REFERENCE) else None
    T_seed, T_ci, T_diff, T_h, T_sur, T_curve, T_ch = [], [], [], [], [], [], []
    keys = ["coverage_%", "timely_%", "timely_lead1_%", "cue_at_onset_%", "false_alarms_per_hour", "time_cue_on_%"]
    for ds, path in DATASETS.items():
        if not os.path.exists(path):
            print(f"{ds}: {path} not found - skipped"); continue
        _, X, meta, _ = pickle.load(open(path, "rb"))
        X = X.reset_index(drop=True); meta = meta.reset_index(drop=True)
        y2 = meta["y2"].to_numpy(); ev = Ev(meta, y2, np.arange(len(meta)))
        print(f"\n=== {ds}: {meta.subject.nunique()} patients, {len(meta) * WIN_S / 3600:.2f} h ===")
        for seed in SEEDS:
            S, ON, ch, _ = run_seed(ds, X, meta, seed)
            T_ch.append(ch)
            W = meta[["subject", "run", "seg", "pos", "y2"]].copy()
            for k, v in S.items():
                W[f"score | {k}"] = v.round(5)
            for k, v in ON.items():
                W[f"on | {k}"] = v.astype(np.int8)
            W.to_csv(f"{OUT}/MX1_per_window_{ds.replace(' ', '_')}_seed{seed}.csv.gz", index=False, compression="gzip")
            for k, v in ON.items():
                c, b = (k.split(" | ") + ["-"])[:2]
                T_seed.append({"dataset": ds, "seed": seed, "controller": c, "budget": b, **pooled(ev.counts(v))})
            pd.DataFrame(T_seed).round(3).to_csv(f"{OUT}/MX1_T1_all_seeds_budgets.csv", index=False)
            if seed != PRIMARY_SEED:
                continue
            if ref is not None:
                r = ref[ref.dataset == ds].reset_index(drop=True)
                print(f"  check: union score equals rerun score: {np.allclose(S['union'].round(5), r['score'].to_numpy(), atol=1e-4)}; "
                      f"primary on-vector equals frozen primary: {(ON['Primary'] == r['on | Controller, floor 0.70'].astype(bool)).mean():.4f}")
            C = {"Primary": ev.counts(ON["Primary"])}
            for c in NEW:
                C[c] = ev.counts(ON[f"{c} | {PRIMARY_BUDGET}"])
            for name, c in C.items():
                ci = boot_ci(c, rng, keys)
                T_ci.append({"dataset": ds, "controller": name, **{k: ci[k][0] for k in keys},
                             **{f"{k}_CI_low": ci[k][1] for k in keys}, **{f"{k}_CI_high": ci[k][2] for k in keys}})
            pairs = [("U*", "Primary"), ("U* (no hold)", "Primary"), ("U*", "U* (no hold)"), ("D", "U*"),
                     ("A", "D"), ("D", "Primary"), ("A", "Primary")]
            Dd = {}
            for a, b in pairs:
                Dd[(a, b)] = boot_diff(C[a], C[b], rng, keys)
                for k in keys:
                    T_diff.append({"dataset": ds, "comparison": f"{a} - {b}", "metric": k, "difference": Dd[(a, b)][k][0],
                                   "CI_low": Dd[(a, b)][k][1], "CI_high": Dd[(a, b)][k][2]})
            for h, r in verdict_h(ds, Dd):
                T_h.append({"dataset": ds, "hypothesis": h, "result": r})
            for c in NEW:
                for r in surrogate(ev, ON[f"{c} | {PRIMARY_BUDGET}"], rng):
                    T_sur.append({"dataset": ds, "controller": c, **r})
            for c, sk in [("U*", "union"), ("D", "fog"), ("A", "comb")]:
                for th in GRID:
                    T_curve.append({"dataset": ds, "controller": c, "theta_on": th[0], "theta_off": th[1],
                                    **pooled(ev.counts(ev.hold(ev.hyst(S[sk], *th))))})
            pd.DataFrame(T_ci).round(3).to_csv(f"{OUT}/MX1_T2_controllers_CI.csv", index=False)
            pd.DataFrame(T_diff).round(3).to_csv(f"{OUT}/MX1_T3_paired_differences.csv", index=False)
            pd.DataFrame(T_h).to_csv(f"{OUT}/MX1_T4_hypotheses.csv", index=False)
            pd.DataFrame(T_sur).round(3).to_csv(f"{OUT}/MX1_T5_surrogate.csv", index=False)
            pd.DataFrame(T_curve).round(3).to_csv(f"{OUT}/MX1_T6_operating_curves.csv", index=False)
            pd.concat(T_ch).to_csv(f"{OUT}/MX1_choices.csv", index=False)
            show = ["controller"] + keys
            print(pd.DataFrame([x for x in T_ci if x["dataset"] == ds])[show].round(1).to_string(index=False))
            print(pd.DataFrame([x for x in T_h if x["dataset"] == ds]).to_string(index=False))
        pd.concat(T_ch).to_csv(f"{OUT}/MX1_choices.csv", index=False)
    if T_seed:
        R = pd.DataFrame(T_seed)
        R = R[(R.budget == str(PRIMARY_BUDGET)) | (R.controller == "Primary")]
        rng_tab = R.groupby(["dataset", "controller"])[keys].agg(["min", "max"]).round(2)
        rng_tab.to_csv(f"{OUT}/MX1_T7_seed_range.csv")
        print("\nRange across seeds (budget 60)\n", rng_tab.to_string())
    print(f"\nDone. Files in {os.path.join(BASE, OUT)}")


if __name__ == "__main__":
    main()
