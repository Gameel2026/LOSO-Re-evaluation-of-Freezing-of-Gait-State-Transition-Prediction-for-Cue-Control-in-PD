import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from fog_pipeline import Config, relabel_four_class
from fog_study import sequences

OUT = "results_precursor"
QUICK = False
warnings.filterwarnings("ignore")
DATASETS = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
            "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")],
            "FoG-STAR": ["results_fogstar_causal"]}
KEYS = {"Freeze index": "freezeindex", "Locomotor-band power": "locomotor", "Freeze-band power": "freezepower",
        "Standard deviation": "stdev"}      
PRE, POST = 16, 8                    


def load(cands):
    for d in cands:
        f = os.path.join(d, "cache_Shank.pkl")
        if os.path.exists(f):
            _, X, meta, _ = pickle.load(open(f, "rb")); return X, meta
    return None, None


def loso_auc(X, y, groups, seed=42):
    score = np.full(len(y), np.nan)
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        if len(np.unique(y[tr])) < 2: continue
        m = RandomForestClassifier(n_estimators=30 if QUICK else 100, class_weight="balanced", n_jobs=-1,
                                   random_state=seed).fit(X[tr], y[tr])
        score[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(score)
    pooled = roc_auc_score(y[ok], score[ok])
    per = [roc_auc_score(y[groups == s], score[groups == s]) for s in np.unique(groups)
           if len(np.unique(y[groups == s])) == 2]
    return pooled, np.median(per), len(per)


def trajectories(X, meta, y4, cols):
    Z = X[cols].to_numpy(float).copy(); g = meta["subject"].to_numpy()
    for s in np.unique(g):                                   
        m = (g == s); base = m & (y4 == 0)
        if base.sum() > 5:                                   
            med = np.median(Z[base], 0); mad = 1.4826 * np.median(np.abs(Z[base] - med), 0) + 1e-9
            Z[m] = np.clip((Z[m] - med) / mad, -10, 10)
    y2 = meta["y2"].to_numpy(); acc = {k: [] for k in range(-PRE, POST)}
    for _, idx in sequences(meta):
        yy = y2[idx]
        for i in np.flatnonzero((yy == 2) & np.r_[True, yy[:-1] != 2]):
            for k in range(-PRE, POST):
                j = i + k
                if 0 <= j < len(idx): acc[k].append(Z[idx[j]])
    return {k: np.nanmean(np.array(v), axis=0) if v else np.full(len(cols), np.nan) for k, v in acc.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = Config(causal_filter=True, sensors=("ank",))
    sep, traj = [], []
    for ds, cands in DATASETS.items():
        X, meta = load(cands)
        if X is None: print(f"{ds}: cache not found - skipped"); continue
        y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w); g = meta["subject"].to_numpy(); F = X.to_numpy(float)
        for name, cls in [("Pre-FoG vs No-FoG", 1), ("FoG vs No-FoG (reference)", 2)]:
            m = np.isin(y4, [0, cls])
            pooled, med, n = loso_auc(F[m], (y4[m] == cls).astype(int), g[m])
            sep.append({"dataset": ds, "comparison": name, "AUC_pooled": pooled, "AUC_median_per_patient": med, "patients": n})
            print(f"  {ds:15s} {name:28s} AUC {pooled:.3f} (median per patient {med:.3f}, n = {n})")
        cols = {lab: [c for c in X.columns if key in c.lower()][:3] for lab, key in KEYS.items()}
        for lab, cl in cols.items():
            if not cl: continue
            t = trajectories(X, meta, y4, cl)
            for k, v in t.items():
                traj.append({"dataset": ds, "feature": lab, "t_s": k * 0.5, "z_mean_over_axes": float(np.nanmean(v))})
    S = pd.DataFrame(sep).round(3); S.to_csv(f"{OUT}/P1_separability.csv", index=False)
    T = pd.DataFrame(traj); T.round(3).to_csv(f"{OUT}/P2_trajectories.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nSeparability, leave-one-subject-out (exploratory)\n", S.to_string(index=False))
    if len(T):
        feats = T.feature.unique(); fig, axes = plt.subplots(1, len(feats), figsize=(3.2 * len(feats), 2.8), sharey=False)
        axes = np.atleast_1d(axes)
        for ax, f in zip(axes, feats):
            for ds, c in zip(DATASETS, ["#1f4e79", "#c0392b", "#2e7d32"]):
                d = T[(T.feature == f) & (T.dataset == ds)]
                if len(d): ax.plot(d.t_s, d.z_mean_over_axes, color=c, label=ds, lw=1.4)
            ax.axvline(0, color="black", lw=0.6, ls="--"); ax.axvspan(-4, 0, color="grey", alpha=0.12)
            ax.set_title(f, fontsize=8); ax.set_xlabel("Time from FoG onset (s)"); ax.set_ylabel("z (vs patient's No-FoG)")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout(); fig.savefig(f"{OUT}/P_trajectories.png", dpi=200)
        print("\nMean z-score in the 4 s before onset (shaded) vs the preceding 4 s:")
        for f in feats:
            for ds in DATASETS:
                d = T[(T.feature == f) & (T.dataset == ds)]
                if len(d):
                    a = d[(d.t_s >= -4) & (d.t_s < 0)].z_mean_over_axes.mean(); b = d[(d.t_s >= -8) & (d.t_s < -4)].z_mean_over_axes.mean()
                    print(f"  {f:22s} {ds:15s} last 4 s {a:+.2f}  earlier 4 s {b:+.2f}  change {a - b:+.2f}")
    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()
