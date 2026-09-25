from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors

# --------------------------------------------------------------------------- #
# 0. Configuration (values from the paper; "ASSUMED" = not reported)
# --------------------------------------------------------------------------- #
NOFOG, PRE, FOG, POST = 0, 1, 2, 3
CLASS_NAMES = {NOFOG: "No-FoG", PRE: "Pre-FoG", FOG: "FoG", POST: "Post-FoG"}


@dataclass
class Config:
    fs: int = 64                         # DAPHNET sampling rate (Hz)
    win_sec: float = 0.5                 # non-overlapping window -> 32 samples
    band: tuple = (0.5, 20.0)            # band-pass cut-offs (Hz)
    filt_order: int = 4                  # Butterworth order (scipy doubles it for band-pass)
    fog_min_sec: float = 0.3             # window labelled FoG if >= 0.3 s of FoG inside
    loco_band: tuple = (0.5, 3.0)        # locomotor band (Hz)
    freeze_band: tuple = (3.0, 8.0)      # freeze band (Hz)
    nfft: int = 256                      # ASSUMED: zero-padding for smoother PSD
    mode_resolution: float = 10.0        # ASSUMED: mg bin for "mode" of a continuous signal
    pre_fog_sec: float = 4.0             # final average pre-FoG length
    post_fog_sec: float = 3.0            # final average post-FoG length
    regression_feats: tuple = ("Spectralpeak_1", "max_2")  # top MI feature per category
    ref_subject: int = 4                 # FoG-free subject used as regression "ground truth"
    thr_k: float = 2.0                   
    n_estimators: int = 100              
    n_splits: int = 10
    test_size: float = 0.30
    smote_k: int = 5                     # ASSUMED (SMOTE default)
    seed: int = 42
    sensors: tuple = ("ank",)            # any of "ank" (shank), "thi" (thigh), "tr" (trunk)

    @property
    def win(self) -> int:
        return int(round(self.fs * self.win_sec))

    @property
    def pre_w(self) -> int:
        return int(round(self.pre_fog_sec / self.win_sec))    # 8 windows

    @property
    def post_w(self) -> int:
        return int(round(self.post_fog_sec / self.win_sec))   # 6 windows


# Final hybrid set (Table 3). Table 3 prints only 22 names; the 23rd is taken
# from Fig. 8 (rank 11: spectralEntropy_1) so that counts match 14 + 9.
# Axis suffix: 1 = horizontal forward (x), 2 = vertical (y), 3 = lateral (z).
HYBRID_23 = [
    # spectral (14)
    "Spectralpeak_1", "Spectralpeak_3", "Meanf_1", "medf_1", "Freezeindex_1",
    "spectralcentroid_1", "spectralEntropy_1", "Locomotorypower_2", "Freezeindex_2",
    "Meanf_2", "medf_2", "Sumpower_2", "Meanf_3", "medf_3",
    # statistical (9)
    "max_2", "max_3", "max_1", "min_1", "mode_1", "RMS_2", "kurtosis_1",
    "Variance_2", "StDev_2",
]

EPS = 1e-12

# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
COLS = ["t_ms", "ank_x", "ank_y", "ank_z", "thi_x", "thi_y", "thi_z",
        "tr_x", "tr_y", "tr_z", "label"]


def load_daphnet(root: str) -> pd.DataFrame:
    """Read every S??R??.txt file (space separated, 11 columns, Table 1)."""
    frames = []
    for path in sorted(glob.glob(os.path.join(root, "S*R*.txt"))):
        name = os.path.basename(path)
        df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
        df["subject"], df["run"] = int(name[1:3]), int(name[4:6])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No DAPHNET files found in {root}")
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Pre-processing: remove label 0, filter, segment, 2-class labels
# --------------------------------------------------------------------------- #
def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True values."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def bandpass(x: np.ndarray, cfg: Config) -> np.ndarray:
    b, a = signal.butter(cfg.filt_order, cfg.band, btype="bandpass", fs=cfg.fs)
    return signal.filtfilt(b, a, x, axis=0)


def segment_windows(df: pd.DataFrame, cfg: Config):
    """
    Filter each contiguous (label != 0) stretch of the shank signal, then cut it
    into non-overlapping 0.5 s windows. Filtering is done on the continuous
    stretch (not per 32-sample window) to avoid edge artefacts.

    Returns
    -------
    windows : (N, 32, 3*len(cfg.sensors)) array  (axes 1-3 = first sensor, 4-6 = second...)
    meta    : DataFrame with subject, run, seg, pos (window order) and y2 (1/2)
    """
    W, min_fog = cfg.win, int(round(cfg.fog_min_sec * cfg.fs))
    wins, meta = [], []
    cols = [f"{s}_{a}" for s in cfg.sensors for a in "xyz"]
    for (subj, run), g in df.groupby(["subject", "run"], sort=True):
        acc = g[cols].to_numpy(float)
        lab = g["label"].to_numpy()
        for seg, (a, b) in enumerate(contiguous_runs(lab != 0)):   # step 3.2.1
            if b - a < 3 * W:                                        # too short to filter
                continue
            xf = bandpass(acc[a:b], cfg)                            # step 3.2.3
            for k in range((b - a) // W):                            # step 3.2.2
                sl = slice(k * W, (k + 1) * W)
                n_fog = int(np.sum(lab[a:b][sl] == 2))
                y2 = 2 if n_fog >= min_fog else 1                   # step 3.2.4
                wins.append(xf[sl])
                meta.append((subj, run, seg, k, y2))
    meta = pd.DataFrame(meta, columns=["subject", "run", "seg", "pos", "y2"])
    return np.stack(wins), meta


# --------------------------------------------------------------------------- #
# 3. Feature extraction (Fig. 4): 14 statistical + 10 spectral per axis
# --------------------------------------------------------------------------- #
def statistical_features(x: np.ndarray, cfg: Config) -> dict:
    vals, cnt = np.unique(np.round(x / cfg.mode_resolution), return_counts=True)
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "mode": vals[cnt.argmax()] * cfg.mode_resolution,
        "min": x.min(),
        "max": x.max(),
        "range": np.ptp(x),
        "harmean": stats.hmean(np.abs(x) + EPS),       # on magnitude (must be > 0)
        "StDev": x.std(ddof=1),
        "Variance": x.var(ddof=1),
        "Mean_Ab": np.mean(np.abs(x - x.mean())),       # mean absolute deviation
        "Med_Ab": np.mean(np.abs(x - np.median(x))),    # avg distance from median
        "kurtosis": stats.kurtosis(x, fisher=False),    # MATLAB-style (non-excess)
        "skewness": stats.skew(x),
        "RMS": np.sqrt(np.mean(x ** 2)),
    }


def spectral_features(x: np.ndarray, cfg: Config) -> dict:
    f, P = signal.periodogram(x, fs=cfg.fs, nfft=cfg.nfft, window="hann",
                              detrend="constant")
    df_ = f[1] - f[0]
    bp = lambda lo, hi: P[(f >= lo) & (f < hi)].sum() * df_
    loco, frz = bp(*cfg.loco_band), bp(*cfg.freeze_band)

    p = P / (P.sum() + EPS)                      # normalised PSD (probability mass)
    mean_f = np.sum(f * p)                       # power-weighted mean frequency
    mag = np.sqrt(P); mag /= mag.sum() + EPS
    centroid = np.sum(f * mag)                   # ASSUMED: magnitude-weighted centroid
    sd = np.sqrt(np.sum((f - mean_f) ** 2 * p)) + EPS
    return {
        "Locomotorypower": loco,
        "Freezepower": frz,
        "Freezeindex": frz / (loco + EPS),        # FI = P(3-8 Hz) / P(0.5-3 Hz)
        "Sumpower": frz + loco,
        "Meanf": mean_f,
        "medf": f[min(np.searchsorted(np.cumsum(P), 0.5 * P.sum()), len(f) - 1)],
        "spectralcentroid": centroid,
        "spectralKurtosis": np.sum((f - mean_f) ** 4 * p) / sd ** 4,
        "spectralEntropy": -np.sum(p * np.log2(p + EPS)) / np.log2(len(p)),
        "Spectralpeak": f[np.argmax(P)],          # dominant frequency
    }


def extract_features(windows: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Return (N, 24*axes) features named like the paper: '<feature>_<axis>'."""
    rows = []
    for w in windows:
        row = {}
        for ax in range(w.shape[1]):
            x = w[:, ax]
            for k, v in statistical_features(x, cfg).items():
                row[f"{k}_{ax + 1}"] = v
            for k, v in spectral_features(x, cfg).items():
                row[f"{k}_{ax + 1}"] = v
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)


# --------------------------------------------------------------------------- #
# 4. Mutual-information ranking (Sec. 3.5 / 4.2)
# --------------------------------------------------------------------------- #
def rank_by_mi(X: pd.DataFrame, y: np.ndarray, seed: int) -> pd.Series:
    mi = mutual_info_classif(X.to_numpy(), y, random_state=seed)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# 5. Regression-based relabelling (Sec. 3.6, Steps 1-6)
# --------------------------------------------------------------------------- #
def _ordered_groups(meta: pd.DataFrame):
    """Yield (subject, positional indices ordered in time) per contiguous stretch."""
    for (subj, _, _), g in meta.groupby(["subject", "run", "seg"], sort=True):
        yield subj, g.sort_values("pos").index.to_numpy()


def regression_scores(X: pd.DataFrame, meta: pd.DataFrame, cfg: Config):
    """Steps 1-2: fit y2 ~ {Spectralpeak_1, Maximum_2}; return predictions + threshold."""
    feats = list(cfg.regression_feats)
    reg = LinearRegression().fit(X[feats], meta["y2"])
    r = reg.predict(X[feats])
    # Step 3: baseline from the FoG-free reference subject (Subject 4 in the paper)
    ref = meta["subject"].eq(cfg.ref_subject).to_numpy()
    if not ref.any():
        ref = meta["y2"].eq(1).to_numpy()
    thr = r[ref].mean() + cfg.thr_k * r[ref].std()
    return r, thr


def estimate_transition_lengths(meta, r, thr, cfg: Config) -> pd.DataFrame:
    """
    Step 4-5: per FoG episode, locate
      TP1 = earliest window before onset from which the regression output stays
            above the No-FoG baseline (progressive rise)       -> pre-FoG length
      TP2 = last above-baseline peak after FoG offset before the output settles
            back to baseline (the 'jerk' to regain gait)       -> post-FoG length
    The paper finds TP1/TP2 by inspecting the curves; this is one operationalisation.
    """
    y2 = meta["y2"].to_numpy()
    recs = []
    for subj, idx in _ordered_groups(meta):
        y, rr = y2[idx], r[idx]
        for s, e in contiguous_runs(y == 2):
            pre, i = 0, s - 1
            while i >= 0 and pre < cfg.pre_w and y[i] == 1 and rr[i] > thr:
                pre += 1; i -= 1
            post = 0
            for j in range(e, min(e + cfg.post_w, len(y))):
                if y[j] != 1:
                    break
                if rr[j] > thr:
                    post = j - e + 1
            recs.append({"subject": subj, "pre_w": pre, "post_w": post})
    return pd.DataFrame(recs)


def relabel_four_class(meta: pd.DataFrame, pre_w, post_w) -> np.ndarray:
    """
    Step 6: assign Pre-FoG / Post-FoG windows around every FoG episode.
    pre_w / post_w: int (global, paper final = 8 / 6 windows) or {subject: int}.
    When episodes are close together, the gap is shared dynamically; pre-FoG gets
    priority (cue activation), giving pre 1-4 s and post 0-3 s as in the paper.
    """
    y2 = meta["y2"].to_numpy()
    y4 = np.where(y2 == 2, FOG, NOFOG)
    get = lambda v, s: v[s] if isinstance(v, dict) else v
    for subj, idx in _ordered_groups(meta):
        P, Q = get(pre_w, subj), get(post_w, subj)
        eps = contiguous_runs(y2[idx] == 2)
        n = len(idx)
        pre_start = []
        for m, (s, _) in enumerate(eps):
            prev_end = eps[m - 1][1] if m else 0
            pre = min(P, s - prev_end)
            y4[idx[s - pre:s]] = PRE
            pre_start.append(s - pre)
        for m, (_, e) in enumerate(eps):
            limit = pre_start[m + 1] if m + 1 < len(eps) else n
            y4[idx[e:e + min(Q, limit - e)]] = POST
    return y4


# --------------------------------------------------------------------------- #
# 6. SMOTE (uses imbalanced-learn if installed, otherwise a minimal version)
# --------------------------------------------------------------------------- #
def smote(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    try:
        from imblearn.over_sampling import SMOTE
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    except ImportError:
        pass
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    Xs, ys = [X], [y]
    for c, n in zip(classes, counts):
        need = counts.max() - n
        if need == 0 or n < 2:
            continue
        Xc = X[y == c]
        nn = NearestNeighbors(n_neighbors=min(k, n - 1) + 1).fit(Xc)
        neigh = nn.kneighbors(Xc, return_distance=False)[:, 1:]
        base = rng.integers(0, n, need)
        mate = neigh[base, rng.integers(0, neigh.shape[1], need)]
        gap = rng.random((need, 1))                     # x_new = x + u * (x_nn - x)
        Xs.append(Xc[base] + gap * (Xc[mate] - Xc[base]))
        ys.append(np.full(need, c))
    return np.vstack(Xs), np.concatenate(ys)


# --------------------------------------------------------------------------- #
# 7. Model + metrics (Table 2)
# --------------------------------------------------------------------------- #
def make_rf(cfg: Config) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=cfg.n_estimators, criterion="gini",
                                  bootstrap=True, n_jobs=-1, random_state=cfg.seed)


def _div(a, b):
    return a / b if b else 0.0


def per_class_metrics(y_true, y_pred, labels=(NOFOG, PRE, FOG, POST)) -> pd.DataFrame:
    """One-vs-rest metrics per class, as in Table 5."""
    rows = {}
    for c in labels:
        t, p = y_true == c, y_pred == c
        tp, tn = float(np.sum(t & p)), float(np.sum(~t & ~p))
        fp, fn = float(np.sum(~t & p)), float(np.sum(t & ~p))
        n = tp + tn + fp + fn
        acc = (tp + tn) / n
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
        rows[CLASS_NAMES[c]] = {
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": _div(2 * prec * rec, prec + rec), "error_rate": 1 - acc,
            "kappa": _div(acc - pe, 1 - pe), "jaccard": _div(tp, tp + fp + fn),
            "mcc": _div(tp * tn - fp * fn,
                        np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
        }
    df = pd.DataFrame(rows).T
    df.loc["Average"] = df.mean()
    return df.round(4)


def weighted_scores(y_true, y_pred) -> dict:
    """Weighted averages (explains accuracy == recall in Table 4)."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted",
                                                 zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p,
            "recall": r, "f1": f}


# --------------------------------------------------------------------------- #
# 8. Evaluation protocols
# --------------------------------------------------------------------------- #
def paper_protocol(X: pd.DataFrame, y4: np.ndarray, feats: list, cfg: Config):
    """Faithful to the paper: SMOTE on all data -> 10-fold stratified CV + 70/30 split."""
    Xb, yb = smote(X[feats].to_numpy(), y4, cfg.smote_k, cfg.seed)
    skf = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed)
    folds = []
    for tr, te in skf.split(Xb, yb):
        m = make_rf(cfg).fit(Xb[tr], yb[tr])
        folds.append(weighted_scores(yb[te], m.predict(Xb[te])))
    cv = pd.DataFrame(folds).mean()

    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=cfg.test_size,
                                          stratify=yb, random_state=cfg.seed)
    pred = make_rf(cfg).fit(Xtr, ytr).predict(Xte)
    return cv, per_class_metrics(yte, pred), confusion_matrix(yte, pred)


def subject_independent(X: pd.DataFrame, y4: np.ndarray, groups: np.ndarray,
                        cfg: Config, k_feats: int = 23):
    """Leave-One-Subject-Out; MI ranking and SMOTE fitted on training folds only."""
    y_true, y_pred = [], []
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        if not np.any(y4[te] != NOFOG):          # skip FoG-free test subjects
            continue
        feats = rank_by_mi(X[tr], y4[tr], cfg.seed).index[:k_feats].tolist()
        Xb, yb = smote(X.loc[tr, feats].to_numpy(), y4[tr], cfg.smote_k, cfg.seed)
        m = make_rf(cfg).fit(Xb, yb)
        y_true.append(y4[te]); y_pred.append(m.predict(X.loc[te, feats].to_numpy()))
    yt, yp = np.concatenate(y_true), np.concatenate(y_pred)
    return weighted_scores(yt, yp), per_class_metrics(yt, yp), confusion_matrix(yt, yp)


# --------------------------------------------------------------------------- #
# 9. Main
# --------------------------------------------------------------------------- #
def main(root: str, cfg: Config = Config()):
    raw = load_daphnet(root)
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg)
    print(f"windows: {len(X)}  features: {X.shape[1]}  FoG windows: {(meta.y2 == 2).sum()}")

    # MI ranking on 2-class labels (Fig. 5) -> top feature per category for regression
    print("Top 2-class MI features:\n", rank_by_mi(X, meta.y2.to_numpy(), cfg.seed).head(6))

    # Relabelling (Sec. 3.6)
    r, thr = regression_scores(X, meta, cfg)
    lengths = estimate_transition_lengths(meta, r, thr, cfg)
    print("Estimated per-subject mean lengths (s):\n",
          (lengths.groupby("subject").mean() * cfg.win_sec).round(2))
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)   # paper final: 4 s / 3 s
    print("4-class distribution:",
          {CLASS_NAMES[c]: int(n) for c, n in zip(*np.unique(y4, return_counts=True))})

    # 4-class MI ranking (Fig. 8) vs. the paper's fixed hybrid set (Table 3)
    mi4 = rank_by_mi(X, y4, cfg.seed)
    print("Top-23 by 4-class MI (this run):", mi4.index[:23].tolist())

    cv, per_cls, cm = paper_protocol(X, y4, HYBRID_23, cfg)
    print("\n[Paper protocol] 10-fold CV (weighted):\n", cv.round(4))
    print(per_cls); print(cm)

    cv2, per_cls2, cm2 = subject_independent(X, y4, meta.subject.to_numpy(), cfg)
    print("\n[LOSO, leakage-free]:", {k: round(v, 4) for k, v in cv2.items()})
    print(per_cls2); print(cm2)


if __name__ == "__main__":
    main(r"D:\dataset\dataset")