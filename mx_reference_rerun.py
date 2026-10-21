import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
BASE = next((c for c in CANDIDATES if os.path.isdir(os.path.join(c, "results_constrained"))), HERE)
os.chdir(BASE)
sys.path.insert(0, BASE)

import fog_improve as fi
from fog_pipeline import relabel_four_class
from fog_study import apply_hysteresis, CUE_ON
from fog_fix3 import Model

OUT = "results_methodsx"
DATASETS = {"DAPHNET": "results_causal", "Multimodal FoG": "results_li_causal", "FoG-STAR": "results_fogstar_causal"}
CHECK = {"DAPHNET": "DAPHNET C0spec", "Multimodal FoG": "Li2021 C0spec"}
FOGSTAR_REF = {"timely_pre_onset_%": 13.861, "detected_%": 67.327}
FLOORS = [None, 0.60, 0.65, 0.70, 0.75, 0.80]
FI_Q = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99]
POW_Q = [None, 0.10, 0.20, 0.30]


def fl_name(f):
    return "no floor" if f is None else f"floor {f:.2f}"


def grids():
    std = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
    flat = [(round(a, 2), round(a, 2)) for a in np.arange(0.30, 0.91, 0.05)]
    wide = [(round(a, 2), round(b, 2)) for a in np.arange(0.05, 0.96, 0.05) for b in np.arange(0.05, a + 1e-9, 0.05)]
    return std, flat, wide


def pick(stats, grid, floor):
    best, best_v = None, (-1, -1.0)
    for th in grid:
        sens, spec, f1 = stats[th]
        v = (1, f1) if (floor is None or spec >= floor) else (0, 0.5 * (sens + spec))
        if v > best_v:
            best, best_v = th, v
    return best, best_v[0] == 1


def rates(y, p):
    t = y.astype(bool)
    sens = p[t].mean() if t.any() else 0.0
    spec = (~p[~t]).mean() if (~t).any() else 0.0
    return sens, spec, f1_score(y, p, zero_division=0)


def freeze_index_rule(X, y_on, tr, te, floor=0.70):
    best, best_v = None, (-1, -1.0)
    for k in (1, 2, 3):
        fcol, pcol = f"Freezeindex_{k}", f"Sumpower_{k}"
        if fcol not in X.columns:
            continue
        fv = X[fcol].to_numpy(float); pv = X[pcol].to_numpy(float) if pcol in X.columns else np.zeros(len(fv))
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0); pv = np.nan_to_num(pv, nan=0.0, posinf=0.0, neginf=0.0)
        for qf in FI_Q:
            tf = np.quantile(fv[tr], qf)
            for qp in POW_Q:
                tp = -np.inf if qp is None else np.quantile(pv[tr], qp)
                p = (fv[tr] >= tf) & (pv[tr] >= tp)
                sens, spec, f1 = rates(y_on[tr], p)
                v = (1, f1) if spec >= floor else (0, 0.5 * (sens + spec))
                if v > best_v:
                    best, best_v = (k, qf, qp, tf, tp), v
    k, qf, qp, tf, tp = best
    fv = np.nan_to_num(X[f"Freezeindex_{k}"].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    pcol = f"Sumpower_{k}"
    pv = np.nan_to_num(X[pcol].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0) if pcol in X.columns else np.zeros(len(fv))
    on_te = (fv[te] >= tf) & (pv[te] >= tp)
    return on_te, fv[te], {"fi_axis": k, "fi_quantile": qf, "power_quantile": qp if qp is not None else 0.0,
                           "fi_threshold": tf, "power_threshold": tp, "fi_feasible": best_v[0] == 1}


def run_dataset(ds, src, cfg):
    fi.SRC = src
    X, meta = fi.load("Shank")
    meta = meta.reset_index(drop=True); X = X.reset_index(drop=True)
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    y_on = np.isin(y4, CUE_ON).astype(int); groups = meta["subject"].to_numpy()
    std, flat, wide = grids()
    union = sorted(set(std) | set(flat) | set(wide))
    variants = {f"Controller, {fl_name(f)}": (std, f) for f in FLOORS}
    variants["No hysteresis, floor 0.70"] = (flat, 0.70)
    variants["No hysteresis, no floor"] = (flat, None)
    variants["Controller, floor 0.70, wide grid"] = (wide, 0.70)
    on = {v: np.zeros(len(y4), bool) for v in variants}
    on["Freeze index, floor 0.70"] = np.zeros(len(y4), bool)
    score = np.zeros(len(y4)); fi_score = np.zeros(len(y4)); choices = []
    for s in np.unique(groups):
        t0 = time.time()
        tr, te = groups != s, groups == s
        feats = fi.rank_mi(X[tr], y_on[tr], cfg.seed).index[:fi.K_FEATS].tolist()
        tr_idx = np.flatnonzero(tr); oof = np.zeros(len(y4))
        for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
            m = Model("RF", cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[tr_idx[a]][feats].to_numpy(), y_on[tr_idx[a]])
            P, cl = m.predict_proba(X.iloc[tr_idx[b]][feats].to_numpy()); oof[tr_idx[b]] = P[:, list(cl).index(1)]
        m = Model("RF", cfg).fit(X.iloc[tr_idx][feats].to_numpy(), y_on[tr_idx])
        P, cl = m.predict_proba(X.loc[te, feats].to_numpy()); score[te] = P[:, list(cl).index(1)]
        stats = {th: rates(y_on[tr], apply_hysteresis(oof, meta, tr, *th)[tr]) for th in union}
        row = {"dataset": ds, "subject": s}
        for name, (grid, floor) in variants.items():
            th, feas = pick(stats, grid, floor)
            on[name][te] = apply_hysteresis(score, meta, te, *th)[te]
            row[f"{name} | on"] = th[0]; row[f"{name} | off"] = th[1]; row[f"{name} | feasible"] = feas
        o, f, info = freeze_index_rule(X, y_on, tr, te)
        on["Freeze index, floor 0.70"][te] = o; fi_score[te] = f
        row.update(info); choices.append(row)
        print(f"  [{ds}] subject {s}: done in {time.time() - t0:.0f} s")
    W = meta[["subject", "run", "seg", "pos", "y2"]].copy()
    W.insert(0, "dataset", ds); W["y4"] = y4; W["score"] = score.round(5); W["fi_score"] = fi_score.round(5)
    for name, v in on.items():
        W[f"on | {name}"] = v.astype(np.int8)
    return W, pd.DataFrame(choices), y4, meta, on


def check(ds, y4, meta, on, cfg):
    s = fi.summarise(y4, on["Controller, floor 0.70"], meta, cfg)
    if ds in CHECK and os.path.exists("results_constrained/B1_summary.csv"):
        b1 = pd.read_csv("results_constrained/B1_summary.csv", index_col=0)
        ref = {k: b1.loc[CHECK[ds], k] for k in ["timely_pre_onset_%", "detected_%", "false_alarms_per_hour", "cue_specificity"]}
    elif ds == "FoG-STAR":
        ref = FOGSTAR_REF
    else:
        ref = {}
    ok = all(abs(s[k] - v) <= 1e-3 + 1e-9 for k, v in ref.items()) if ref else False
    print(f"Reproduction check {ds}: {'YES' if ok else 'NO'}  " + ", ".join(f"{k}: {s[k]:.3f} vs {v:.3f}" for k, v in ref.items()))
    return ok


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    cfg = fi.cfg_(); allW, allC, summ, checks = [], [], [], {}
    for ds, src in DATASETS.items():
        if not os.path.exists(os.path.join(src, "cache_Shank.pkl")):
            print(f"{ds}: {src}/cache_Shank.pkl not found - skipped"); continue
        print(f"\n=== {ds} ===")
        W, C, y4, meta, on = run_dataset(ds, src, cfg)
        allW.append(W); allC.append(C)
        pd.concat(allW).to_csv(f"{OUT}/MX_per_window.csv.gz", index=False, compression="gzip")
        pd.concat(allC).to_csv(f"{OUT}/MX_choices.csv", index=False)
        checks[ds] = check(ds, y4, meta, on, cfg)
        for name, v in on.items():
            summ.append({"dataset": ds, "strategy": name, **fi.summarise(y4, v, meta, cfg)})
        pd.DataFrame(summ).round(3).to_csv(f"{OUT}/MX_quick_summary.csv", index=False)
    if not summ:
        print(f"No caches found under {BASE}"); return
    S = pd.DataFrame(summ)
    cols = ["dataset", "strategy", "cue_specificity", "timely_pre_onset_%", "detected_%", "false_alarms_per_hour"]
    print("\n", S[cols].round(2).to_string(index=False))
    print("\nReproduction:", checks)
    print(f"\nSaved {OUT}/MX_per_window.csv.gz, MX_choices.csv, MX_quick_summary.csv in {BASE}")


if __name__ == "__main__":
    main()
