import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from fog_pipeline import Config, relabel_four_class, rank_by_mi, smote, make_rf
from fog_study import CUE_ON
import fog_improve as fi
from fog_stats import holm

OUT = "results_auprc"
QUICK = False
# where the feature caches are (first folder that exists is used)
CANDIDATES = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
              "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")],
              "FoG-STAR": ["results_fogstar_causal"]}


def find_cache(ds):
    for d in CANDIDATES[ds]:
        f = os.path.join(d, "cache_Shank.pkl")
        if os.path.exists(f):
            return f
    raise FileNotFoundError(f"cache_Shank.pkl for {ds} not found in {CANDIDATES[ds]}")


def loso_scores(X, y4, meta, cfg):
    y_on = np.isin(y4, CUE_ON).astype(int)
    groups = meta["subject"].to_numpy()
    sA, sB = np.zeros(len(y4)), np.zeros(len(y4))
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        # A) binary controller model (same as the final controller)
        fa = fi.rank_mi(X[tr], y_on[tr], cfg.seed).index[:23].tolist()
        m = fi.Model("RF", cfg).fit(X.loc[tr, fa].to_numpy(), y_on[tr])
        P, cl = m.predict_proba(X.loc[te, fa].to_numpy()); sA[te] = P[:, list(cl).index(1)]
        # B) four-class random forest (P3)
        fb = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:23].tolist()
        Xb, yb = smote(X.loc[tr, fb].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        rf = make_rf(cfg).fit(Xb, yb)
        P = rf.predict_proba(X.loc[te, fb].to_numpy()); sB[te] = 1 - P[:, list(rf.classes_).index(0)]
        print(f"  subject {s:02d} done")
    return y_on, sA, sB


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 220)
    cfg = Config(causal_filter=True, n_estimators=30 if QUICK else 100)
    summ, per, tests = [], [], []
    fig, axes = plt.subplots(1, len(CANDIDATES), figsize=(16, 4.5))
    for ax, ds in zip(axes, CANDIDATES):
        print(f"\n=== {ds} ===")
        _, X, meta, _ = pickle.load(open(find_cache(ds), "rb"))
        y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
        y, sA, sB = loso_scores(X, y4, meta, cfg)
        prev = y.mean()
        for name, sc in [("Proposed controller (binary RF)", sA), ("Four-class RF (P3)", sB)]:
            summ.append({"dataset": ds, "model": name, "prevalence (chance AUPRC)": prev,
                         "AUPRC pooled": average_precision_score(y, sc), "AUROC pooled": roc_auc_score(y, sc)})
            pr, rc, _ = precision_recall_curve(y, sc); ax.plot(rc, pr, label=f"{name} (AUPRC {summ[-1]['AUPRC pooled']:.3f})")
        ax.axhline(prev, ls="--", color="grey", label=f"Chance ({prev:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(ds); ax.legend(fontsize=8)
        rows = []
        for s in sorted(meta["subject"].unique()):
            m = meta["subject"].eq(s).to_numpy()
            if y[m].min() == y[m].max():          
                continue
            rows.append({"dataset": ds, "subject": s, "prevalence": y[m].mean(),
                         "AUPRC_A": average_precision_score(y[m], sA[m]), "AUPRC_B": average_precision_score(y[m], sB[m]),
                         "AUROC_A": roc_auc_score(y[m], sA[m]), "AUROC_B": roc_auc_score(y[m], sB[m])})
        t = pd.DataFrame(rows); per.append(t)
        fam = []
        for met in ["AUPRC", "AUROC"]:
            d = t[f"{met}_A"] - t[f"{met}_B"]
            fam.append({"dataset": ds, "metric": met, "n": len(t), "median_proposed": t[f"{met}_A"].median(),
                        "median_fourclass": t[f"{met}_B"].median(), "proposed_better": f"{int((d > 0).sum())}/{len(d)}",
                        "p_raw": wilcoxon(t[f"{met}_A"], t[f"{met}_B"]).pvalue})
        fam = pd.DataFrame(fam); fam["p_holm"] = holm(fam["p_raw"].to_numpy()); tests.append(fam)
    fig.tight_layout(); fig.savefig(f"{OUT}/A_pr_curves.png", dpi=300); plt.show()
    s = pd.DataFrame(summ).round(4); s.to_csv(f"{OUT}/A1_summary.csv", index=False)
    pd.concat(per).round(4).to_csv(f"{OUT}/A2_per_patient.csv", index=False)
    t = pd.concat(tests).round(4); t.to_csv(f"{OUT}/A3_tests.csv", index=False)
    print("\nSummary\n", s); print("\nPer-patient tests\n", t)
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()