import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import fog_improve as fi
from fog_fix3 import Model
from fog_study import apply_hysteresis, CUE_ON
from fog_pipeline import relabel_four_class
from fog_floor_sensitivity import select

OUT = "results_nested"
CKPT = os.path.join(OUT, "checkpoints")
QUICK = False                      
K_GRID = [10, 23, 40, 72]
FLOOR, K_PRIMARY = 0.70, 23
BASE = r"D:\New research fog work"
DATASETS = {"DAPHNET": ["results_causal", os.path.join(BASE, "results_causal")],
            "Multimodal FoG": ["results_li_causal", os.path.join(BASE, "results_li_causal")],
            "FoG-STAR": ["results_fogstar_causal", os.path.join(BASE, "results_fogstar_causal")]}
REF = {"DAPHNET": (30.085, 89.407, 127.237), "Multimodal FoG": (32.343, 92.079, 96.022), "FoG-STAR": (13.861, 67.327, 95.272)}
warnings.filterwarnings("ignore")


def find_src(cands):
    for d in cands:
        if os.path.exists(os.path.join(d, "cache_Shank.pkl")):
            return d
    raise FileNotFoundError(f"cache_Shank.pkl not found in {cands}")


def criterion(oof, y_on, meta, tr, th):
    from sklearn.metrics import f1_score
    t_ = y_on[tr].astype(bool); p_ = apply_hysteresis(oof, meta, tr, *th)[tr]
    sens, spec = p_[t_].mean(), (~p_[~t_]).mean()
    return (1, f1_score(y_on[tr], p_, zero_division=0)) if spec >= FLOOR else (0, 0.5 * (sens + spec))


def fold(X, y_on, meta, cfg, s, grid):
    groups = meta["subject"].to_numpy(); tr, te = groups != s, groups == s; tr_idx = np.flatnonzero(tr)
    ranked = fi.rank_mi(X[tr], y_on[tr], cfg.seed).index.tolist()
    ks = [min(k, X.shape[1]) for k in K_GRID]
    oof = {k: np.zeros(len(y_on)) for k in ks}
    for a, b in GroupKFold(3).split(tr_idx, groups=groups[tr_idx]):
        for k in ks:
            f = ranked[:k]
            m = Model("RF", cfg, n_trees=min(cfg.n_estimators, 50)).fit(X.iloc[tr_idx[a]][f].to_numpy(), y_on[tr_idx[a]])
            P, cl = m.predict_proba(X.iloc[tr_idx[b]][f].to_numpy()); oof[k][tr_idx[b]] = P[:, list(cl).index(1)]
    th = {k: select(oof[k], y_on, meta, tr, grid, FLOOR)[0] for k in ks}
    val = {k: criterion(oof[k], y_on, meta, tr, th[k]) for k in ks}
    k_star = max(ks, key=lambda k: (val[k], -k))                     # best criterion; ties -> fewer features
    res = {"subject": s, "k_selected": k_star, "theta_on": th[k_star][0], "theta_off": th[k_star][1],
           "theta_on_23": th[min(K_PRIMARY, X.shape[1])][0], "theta_off_23": th[min(K_PRIMARY, X.shape[1])][1]}
    for tag, k in [("nested", k_star), ("fixed23", min(K_PRIMARY, X.shape[1]))]:
        f = ranked[:k]
        m = Model("RF", cfg).fit(X.iloc[tr_idx][f].to_numpy(), y_on[tr_idx])
        P, cl = m.predict_proba(X.loc[te, f].to_numpy()); score = np.zeros(len(y_on)); score[te] = P[:, list(cl).index(1)]
        res[tag] = apply_hysteresis(score, meta, te, *th[k])[te]
    return res


def main():
    os.makedirs(CKPT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    fi.QUICK = QUICK; cfg = fi.cfg_()
    grid = [(round(a, 2), round(b, 2)) for a in np.arange(0.30, 0.91, 0.05) for b in np.arange(0.10, a + 1e-9, 0.10)]
    summ, sur, per, tests, choices = [], [], [], [], []
    for ds, cands in DATASETS.items():
        fi.SRC = find_src(cands)
        X, meta = fi.load("Shank")
        y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w); y_on = np.isin(y4, CUE_ON).astype(int)
        groups = meta["subject"].to_numpy(); subjects = np.unique(groups)
        if QUICK:
            keep = np.isin(groups, subjects[:5]); X = X[keep].reset_index(drop=True); meta = meta[keep].reset_index(drop=True)
            y4, y_on, groups = y4[keep], y_on[keep], groups[keep]; subjects = np.unique(groups)
        on = {"nested": np.zeros(len(y4), bool), "fixed23": np.zeros(len(y4), bool)}
        for s in subjects:
            ck = os.path.join(CKPT, f"{ds.replace(' ', '_')}_S{int(s):02d}{'_quick' if QUICK else ''}.pkl")
            if os.path.exists(ck):
                r = pd.read_pickle(ck); print(f"  [{ds}] subject {int(s):02d}: loaded from checkpoint")
            else:
                t0 = time.time(); r = fold(X, y_on, meta, cfg, s, grid); pd.to_pickle(r, ck)
                print(f"  [{ds}] subject {int(s):02d}: {r['k_selected']} features, on={r['theta_on']} off={r['theta_off']} "
                      f"({time.time() - t0:.0f} s, saved)")
            te = groups == s
            on["nested"][te] = r["nested"]; on["fixed23"][te] = r["fixed23"]
            choices.append({"dataset": ds, **{k: v for k, v in r.items() if k not in ("nested", "fixed23")}})
        names = {"fixed23": "Controller, 23 features (primary)", "nested": "Controller, nested feature number"}
        table = []
        for key, name in names.items():
            summ.append({"dataset": ds, "controller": name, **fi.summarise(y4, on[key], meta, cfg)})
            sg = fi.surrogate(y4, on[key], meta, cfg)
            for m in ["timely_pre_onset_%", "detected_%"]:
                sur.append({"dataset": ds, "controller": name, "metric": m, **sg.loc[m].to_dict()})
            table.append(fi.per_subject(name, y4, on[key], meta, cfg))
        table = pd.concat(table, ignore_index=True)
        t = fi.tests(table, [(names["nested"], names["fixed23"])]); t.insert(0, "dataset", ds); tests.append(t)
        table.insert(0, "dataset", ds); per.append(table)
    S, G, T, Cc = pd.DataFrame(summ), pd.DataFrame(sur), pd.concat(tests), pd.DataFrame(choices)
    S.round(4).to_csv(f"{OUT}/N1_summary.csv", index=False); G.round(4).to_csv(f"{OUT}/N2_surrogate.csv", index=False)
    pd.concat(per).round(4).to_csv(f"{OUT}/N3_per_subject.csv", index=False); T.round(4).to_csv(f"{OUT}/N4_tests.csv", index=False)
    Cc.to_csv(f"{OUT}/N5_choices.csv", index=False)
    cols = ["dataset", "controller", "cue_specificity", "balanced_acc", "timely_pre_onset_%", "detected_%",
            "median_lead_time_s", "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\nSummary\n", S[cols].round(3).to_string(index=False))
    print("\nSurrogate test\n", G.round(3).to_string(index=False))
    print("\nPer-patient tests, nested vs 23 features (Holm within each dataset)\n", T.round(4).to_string(index=False))
    print("\nNumber of features chosen per fold\n", Cc.groupby("dataset")["k_selected"].value_counts().unstack(fill_value=0).to_string())
    if not QUICK:
        for ds, (tim, det, fa) in REF.items():
            r = S[(S.dataset == ds) & (S.controller.str.contains("primary"))].iloc[0]
            ok = abs(r["timely_pre_onset_%"] - tim) < 0.01 and abs(r["detected_%"] - det) < 0.01 and abs(r["false_alarms_per_hour"] - fa) < 0.01
            print(f"Check {ds}: 23-feature controller reproduces the manuscript -> {'YES' if ok else 'NO (please report)'}")
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
