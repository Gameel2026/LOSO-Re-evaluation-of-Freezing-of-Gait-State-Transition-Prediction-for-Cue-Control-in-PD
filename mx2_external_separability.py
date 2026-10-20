import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

OUT = "results_mx2_external_separability"
SCORE = "score | M"
N_BOOT = 5000
FILES = {"DAPHNET": "results_mx2/MX2_per_window_DAPHNET_seed42.csv.gz",
         "Multimodal FoG": "results_mx2/MX2_per_window_Multimodal_FoG_seed42.csv.gz",
         "FoG-STAR": "results_mx2/MX2_per_window_FoG-STAR_seed42.csv.gz",
         "tdcsfog (lab)": "results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz",
         "defog (home)": "results_mx2_external/EV_per_window_defog_seed42.csv.gz"}
ROLE = {"DAPHNET": "development", "Multimodal FoG": "development", "FoG-STAR": "development",
        "tdcsfog (lab)": "external", "defog (home)": "external"}


def window_sets(W):
    ev = m1.Ev(W, W["y2"].to_numpy(), np.arange(len(W)))
    pre1 = np.zeros(ev.n, bool); last = np.zeros(ev.n, bool)
    for s, a, pe in zip(ev.es, ev.ea, ev.pe):
        lo = a if pe < 0 else max(a, pe + m1.POST_W)
        for i in range(max(lo, s - m1.PRE_W), s - 1):
            pre1[i] = True
        if s - 1 >= lo:
            last[s - 1] = True
    neg = ev.outside & ~ev.fog
    return ev, {"Pre-onset, lead >= 1 s": pre1 & ~ev.fog, "Last window before onset": last & ~ev.fog, "FoG (reference)": ev.fog}, neg


def pair_counts(sc, y, g, pats):
    neg = {q: np.sort(sc[(g == q) & (y == 0)]) for q in pats}
    C = np.zeros((len(pats), len(pats)))
    for i, p in enumerate(pats):
        sp = sc[(g == p) & (y == 1)]
        for j, q in enumerate(pats):
            n = neg[q]
            C[i, j] = np.searchsorted(n, sp, "left").sum() + 0.5 * (np.searchsorted(n, sp, "right") - np.searchsorted(n, sp, "left")).sum()
    P = np.array([((g == p) & (y == 1)).sum() for p in pats], float)
    N = np.array([((g == p) & (y == 0)).sum() for p in pats], float)
    return C, P, N


def auc_rows(name, W, rng):
    ev, pos_sets, neg = window_sets(W)
    s = W[SCORE].to_numpy(); subj = ev.subj; rows = []
    for label, pos in pos_sets.items():
        m = pos | neg; y = pos[m].astype(int); sc = s[m]; g = subj[m]
        auc = roc_auc_score(y, sc)
        pats = np.unique(g)
        both = [p for p in pats if 0 < y[g == p].sum() < (g == p).sum()]
        per = [roc_auc_score(y[g == p], sc[g == p]) for p in both]
        C, P, N = pair_counts(sc, y, g, pats)
        w = np.stack([np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)) for _ in range(N_BOOT)]).astype(float)
        den = (w @ P) * (w @ N); ok = den > 0
        boots = np.einsum("bi,ij,bj->b", w[ok], C, w[ok]) / den[ok]
        rows.append({"cohort": name, "role": ROLE[name], "comparison": f"{label} vs No-FoG", "positive_windows": int(y.sum()),
                     "negative_windows": int((1 - y).sum()), "patients_with_both_classes": len(both), "AUC_pooled": auc,
                     "AUC_CI_low": np.percentile(boots, 2.5), "AUC_CI_high": np.percentile(boots, 97.5),
                     "AUC_median_per_patient": float(np.median(per)), "boot": boots})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(42)
    rows = []
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"{name}: {f} not found - skipped", flush=True); continue
        rows += auc_rows(name, pd.read_csv(f), rng)
        print(f"{name}: done", flush=True)
    R = pd.DataFrame(rows)
    R.drop(columns="boot").round(3).to_csv(os.path.join(OUT, "EVSEP_T1_separability.csv"), index=False)
    D = []
    key = "Pre-onset, lead >= 1 s vs No-FoG"
    a = R[(R.cohort == "tdcsfog (lab)") & (R.comparison == key)]
    b = R[(R.cohort == "defog (home)") & (R.comparison == key)]
    if len(a) and len(b):
        n = min(len(a.boot.iloc[0]), len(b.boot.iloc[0]))
        d = a.boot.iloc[0][:n] - b.boot.iloc[0][:n]
        D.append({"comparison": "AUC lead >= 1 s, tdcsfog minus defog", "difference": a.AUC_pooled.iloc[0] - b.AUC_pooled.iloc[0],
                  "CI_low": np.percentile(d, 2.5), "CI_high": np.percentile(d, 97.5)})
    pd.DataFrame(D).round(3).to_csv(os.path.join(OUT, "EVSEP_T2_external_difference.csv"), index=False)
    print("\n", R.drop(columns="boot").round(3).to_string(index=False))
    print("\n", pd.DataFrame(D).round(3).to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
