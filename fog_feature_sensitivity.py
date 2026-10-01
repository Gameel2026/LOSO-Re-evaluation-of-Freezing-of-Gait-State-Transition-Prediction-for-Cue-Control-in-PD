import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, balanced_accuracy_score

import fog_improve as fi
from fog_pipeline import relabel_four_class, rank_by_mi, smote, make_rf

OUT = "results_features"
CKPT = os.path.join(OUT, "checkpoints")          
QUICK = False                          
K_VALUES = [10, 23, 40, 72]
FLOOR_OBJECTIVE = "f1_spec"            
DATASETS = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
            "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")]}
warnings.filterwarnings("ignore")


def find_src(cands):
    for d in cands:
        if os.path.exists(os.path.join(d, "cache_Shank.pkl")):
            return d
    raise FileNotFoundError(f"cache_Shank.pkl not found in {cands}")


def fourclass_loso(X, y4, groups, cfg, k):
    """Four-class RF under LOSO with in-fold MI selection of k features and in-fold SMOTE (P3)."""
    pred = np.full(len(y4), -1)
    subjects = np.unique(groups)[:5] if QUICK else np.unique(groups)
    for s in subjects:
        tr, te = groups != s, groups == s
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:k].tolist()
        Xb, yb = smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        pred[te] = make_rf(cfg).fit(Xb, yb).predict(X.loc[te, feats].to_numpy())
    m = pred >= 0
    rec = recall_score(y4[m], pred[m], labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {"accuracy": accuracy_score(y4[m], pred[m]), "balanced_acc": balanced_accuracy_score(y4[m], pred[m]),
            "PreFoG_recall": rec[1], "PostFoG_recall": rec[3], "transition_recall": (rec[1] + rec[3]) / 2}


def main():
    os.makedirs(CKPT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    fi.QUICK = QUICK; cfg = fi.cfg_()
    four, ctrl, sur, tests = [], [], [], []
    for ds, cands in DATASETS.items():
        fi.SRC = find_src(cands)
        X, meta = fi.load("Shank")
        y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w); groups = meta["subject"].to_numpy()
        if QUICK:
            keep = np.isin(groups, np.unique(groups)[:5]); X, meta, y4, groups = X[keep], meta[keep].reset_index(drop=True), y4[keep], groups[keep]
            X = X.reset_index(drop=True)
        table = []
        for k in K_VALUES:
            kk = min(k, X.shape[1]); t0 = time.time()
            ck = os.path.join(CKPT, f"{ds.replace(' ', '_')}_k{kk}{'_quick' if QUICK else ''}.pkl")
            if os.path.exists(ck):                                   
                part = pd.read_pickle(ck)
                print(f"  [{ds}] {kk} features: loaded from checkpoint")
            else:
                name = f"Controller, {kk} features"
                fc = {"dataset": ds, "n_features": kk, **fourclass_loso(X, y4, groups, cfg, kk)}
                fi.K_FEATS = kk
                on, _ = fi.controller(X, y4, meta, cfg, ("RF",), FLOOR_OBJECTIVE, f"{ds} k={kk}")
                sg = fi.surrogate(y4, on, meta, cfg)
                part = {"four": fc, "ctrl": {"dataset": ds, "n_features": kk, **fi.summarise(y4, on, meta, cfg)},
                        "sur": [{"dataset": ds, "n_features": kk, "metric": m, **sg.loc[m].to_dict()}
                                for m in ["timely_pre_onset_%", "detected_%"]],
                        "table": fi.per_subject(name, y4, on, meta, cfg)}
                pd.to_pickle(part, ck)                                  
                print(f"  [{ds}] {kk} features done and saved ({time.time() - t0:.0f} s)")
            four.append(part["four"]); ctrl.append(part["ctrl"]); sur.extend(part["sur"]); table.append(part["table"])
        fi.K_FEATS = 23
        table = pd.concat(table, ignore_index=True)
        comps = [(f"Controller, {min(k, X.shape[1])} features", "Controller, 23 features") for k in K_VALUES if k != 23]
        t = fi.tests(table, comps); t.insert(0, "dataset", ds); tests.append(t)
    F = pd.DataFrame(four).round(4); C = pd.DataFrame(ctrl).round(4); S = pd.DataFrame(sur).round(4); T = pd.concat(tests).round(4)
    F.to_csv(f"{OUT}/FK1_fourclass.csv", index=False); C.to_csv(f"{OUT}/FK2_controller.csv", index=False)
    S.to_csv(f"{OUT}/FK3_surrogate.csv", index=False); T.to_csv(f"{OUT}/FK4_tests.csv", index=False)
    cols = ["dataset", "n_features", "cue_specificity", "balanced_acc", "timely_pre_onset_%", "detected_%",
            "median_lead_time_s", "false_alarms_per_hour", "median_cue_off_delay_s"]
    print("\n(A) Four-class RF under LOSO (P3)\n", F.to_string(index=False))
    print("\n(B) Proposed controller (specificity floor 0.70)\n", C[[c for c in cols if c in C.columns]].round(3).to_string(index=False))
    print("\nSurrogate test\n", S.round(3).to_string(index=False))
    print("\nPer-patient tests against 23 features (Holm within each comparison)\n", T.to_string(index=False))
    if not QUICK:
        ref = {"DAPHNET": (0.105, 30.085, 89.407, 127.237), "Multimodal FoG": (0.111, 32.343, 92.079, 96.022)}
        for ds, (tr, tim, det, fa) in ref.items():
            f = F[(F.dataset == ds) & (F.n_features == 23)].iloc[0]; c = C[(C.dataset == ds) & (C.n_features == 23)].iloc[0]
            ok = abs(f.transition_recall - tr) < 0.006 and abs(c["timely_pre_onset_%"] - tim) < 0.01 and abs(c["detected_%"] - det) < 0.01 and abs(c.false_alarms_per_hour - fa) < 0.01
            print(f"Check {ds}: 23 features reproduce the manuscript -> {'YES' if ok else 'NO (please report)'}")
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
