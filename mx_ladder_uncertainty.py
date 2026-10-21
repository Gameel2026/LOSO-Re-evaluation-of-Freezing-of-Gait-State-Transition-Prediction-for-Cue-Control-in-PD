import os
import sys
import pickle
import dataclasses
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
BASE = next((c for c in CANDIDATES if os.path.isdir(os.path.join(c, "results_causal"))), HERE)
os.chdir(BASE)
sys.path.insert(0, BASE)
import fog_pipeline as fp

OUT = "results_ladder_uncertainty"
DATASETS = {"DAPHNET": "results_causal", "Multimodal FoG": "results_li_causal", "FoG-STAR": "results_fogstar_causal"}
SEEDS = [42, 1, 2, 3, 4]
N_BOOT = 5000
PROTOCOLS = ["P0 Original", "P1 SMOTE in folds", "P2 LOSO fixed features", "P3 LOSO in-fold features"]
P0R = "P0 Original (real windows)"
CONTRASTS = [(P0R, "P1 SMOTE in folds"), ("P1 SMOTE in folds", "P2 LOSO fixed features"), ("P1 SMOTE in folds", "P3 LOSO in-fold features"),
             ("P3 LOSO in-fold features", "P2 LOSO fixed features")]


def metrics_from_cm(cm):
    n = cm.sum(); sup = cm.sum(1); pred = cm.sum(0); tp = np.diag(cm)
    rec = np.divide(tp, sup, out=np.zeros(4), where=sup > 0); prec = np.divide(tp, pred, out=np.zeros(4), where=pred > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0); w = sup / max(n, 1)
    return {"accuracy": tp.sum() / max(n, 1), "f1": (w * f1).sum(), "PreFoG_recall": rec[1], "PostFoG_recall": rec[3],
            "transition_recall": 0.5 * (rec[1] + rec[3])}


def cm(y, p):
    m = np.zeros((4, 4)); np.add.at(m, (y, p), 1); return m


def window_cv(Xf, y4, cfg, smote_before):
    if smote_before:
        Xf, y = fp.smote(Xf, y4, cfg.smote_k, cfg.seed)
    else:
        y = y4
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed).split(Xf, y):
        Xtr, ytr = (Xf[tr], y[tr]) if smote_before else fp.smote(Xf[tr], y[tr], cfg.smote_k, cfg.seed)
        pred[te] = fp.make_rf(cfg).fit(Xtr, ytr).predict(Xf[te])
    return y, pred


def loso(X, y4, groups, cfg, in_fold):
    pred = np.empty_like(y4)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        feats = fp.rank_by_mi(X[tr], y4[tr], cfg.seed).index[:23].tolist() if in_fold else fp.HYBRID_23
        Xb, yb = fp.smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        m = fp.make_rf(cfg).fit(Xb, yb)
        if in_fold:
            P = np.full((int(te.sum()), 4), 1e-6); P[:, m.classes_] += m.predict_proba(X.loc[te, feats].to_numpy())
            pred[te] = P.argmax(1)
        else:
            pred[te] = m.predict(X.loc[te, feats].to_numpy())
    return pred


def per_patient_cm(y, p, groups):
    subj = np.unique(groups)
    return subj, np.stack([cm(y[groups == s], p[groups == s]) for s in subj])


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    T1, T2, T3, CHK = [], [], [], []
    for ds, src in DATASETS.items():
        path = os.path.join(src, "cache_Shank.pkl")
        if not os.path.exists(path):
            print(f"{ds}: {path} not found - skipped"); continue
        cfg0, X, meta, y4 = pickle.load(open(path, "rb"))
        X = X.reset_index(drop=True); meta = meta.reset_index(drop=True); groups = meta.subject.to_numpy(); Xf = X[fp.HYBRID_23].to_numpy()
        for seed in SEEDS:
            ck = os.path.join(OUT, f"ckpt_{ds.replace(' ', '_')}_seed{seed}.pkl")
            if os.path.exists(ck):
                res = pickle.load(open(ck, "rb"))
            else:
                cfg = dataclasses.replace(cfg0, seed=seed); res = {}
                print(f"[{ds}] seed {seed}: P0", flush=True); res["P0 Original"] = window_cv(Xf, y4, cfg, True)
                print(f"[{ds}] seed {seed}: P1", flush=True); res["P1 SMOTE in folds"] = window_cv(Xf, y4, cfg, False)
                print(f"[{ds}] seed {seed}: P2", flush=True); res["P2 LOSO fixed features"] = (y4, loso(X, y4, groups, cfg, False))
                print(f"[{ds}] seed {seed}: P3", flush=True); res["P3 LOSO in-fold features"] = (y4, loso(X, y4, groups, cfg, True))
                pickle.dump(res, open(ck, "wb"))
            for pr in PROTOCOLS:
                y, p = res[pr]
                T1.append({"dataset": ds, "seed": seed, "protocol": pr, **metrics_from_cm(cm(y, p))})
            if seed == 42 and os.path.exists(os.path.join(src, "E1_protocols.csv")):
                e1 = pd.read_csv(os.path.join(src, "E1_protocols.csv"), index_col=0)
                for pr in PROTOCOLS:
                    if pr in e1.index:
                        y, p = res[pr]; mine = metrics_from_cm(cm(y, p))
                        CHK.append({"dataset": ds, "protocol": pr, "registered_transition_recall": e1.loc[pr, "transition_recall"],
                                    "rerun_transition_recall": round(mine["transition_recall"], 4),
                                    "registered_f1": e1.loc[pr, "f1"], "rerun_f1": round(mine["f1"], 4)})
            rng = np.random.default_rng(seed); PP = {}
            y0, p0 = res["P0 Original"]
            if not np.array_equal(y0[:len(y4)], y4):
                sys.exit("P0: original windows are not the first rows of the oversampled set")
            subj, PP[P0R] = per_patient_cm(y4, p0[:len(y4)], groups)
            for pr in PROTOCOLS[1:]:
                y, p = res[pr]; subj, C = per_patient_cm(y, p, groups); PP[pr] = C
            n = len(subj); idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
            for pr, C in PP.items():
                B = np.array([list(metrics_from_cm(C[i].sum(0)).values()) for i in idx]); lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
                est = metrics_from_cm(C.sum(0))
                T2.append({"dataset": ds, "seed": seed, "protocol": pr, "patients": n,
                           **{f"{k}{s}": v for j, k in enumerate(est) for s, v in (("", est[k]), ("_CI_low", lo[j]), ("_CI_high", hi[j]))}})
            for a, b in CONTRASTS:
                B = np.array([np.array(list(metrics_from_cm(PP[a][i].sum(0)).values())) - np.array(list(metrics_from_cm(PP[b][i].sum(0)).values())) for i in idx])
                lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
                est = np.array(list(metrics_from_cm(PP[a].sum(0)).values())) - np.array(list(metrics_from_cm(PP[b].sum(0)).values()))
                keys = list(metrics_from_cm(PP[a].sum(0)).keys())
                T3.append({"dataset": ds, "seed": seed, "contrast": f"{a} minus {b}",
                           **{f"{k}{s}": v for j, k in enumerate(keys) for s, v in (("", est[j]), ("_CI_low", lo[j]), ("_CI_high", hi[j]))}})
            pd.DataFrame(T1).round(4).to_csv(os.path.join(OUT, "LAD_T1_by_seed.csv"), index=False)
    R1 = pd.DataFrame(T1)
    spread = R1.groupby(["dataset", "protocol"], sort=False)[["f1", "transition_recall"]].agg(["mean", "min", "max"]).round(4)
    spread.columns = [f"{a}_{b}" for a, b in spread.columns]; spread = spread.reset_index()
    spread.to_csv(os.path.join(OUT, "LAD_T0_seed_spread.csv"), index=False)
    pd.DataFrame(T2).round(4).to_csv(os.path.join(OUT, "LAD_T2_patient_bootstrap.csv"), index=False)
    pd.DataFrame(T3).round(4).to_csv(os.path.join(OUT, "LAD_T3_paired_differences.csv"), index=False)
    pd.DataFrame(CHK).to_csv(os.path.join(OUT, "LAD_T4_seed42_reproduction.csv"), index=False)
    print("\nSeed-42 reproduction\n", pd.DataFrame(CHK).to_string(index=False))
    print("\nSpread across seeds\n", spread.to_string(index=False))
    T3d = pd.DataFrame(T3)
    if len(T3d):
        print("\nPaired differences (patient bootstrap)\n", T3d[["dataset", "seed", "contrast", "transition_recall", "transition_recall_CI_low", "transition_recall_CI_high"]].round(4).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(BASE, OUT)}")


if __name__ == "__main__":
    main()
