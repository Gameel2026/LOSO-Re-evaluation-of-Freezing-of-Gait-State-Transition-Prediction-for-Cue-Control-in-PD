import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mx_new_controllers as m1

OUT = "results_onset_profile"
SCORE = "score | M"
N_BOOT = 5000
OFFSETS = list(range(-m1.PRE_W, 5))
FILES = {"DAPHNET": "results_mx2/MX2_per_window_DAPHNET_seed42.csv.gz",
         "Multimodal FoG": "results_mx2/MX2_per_window_Multimodal_FoG_seed42.csv.gz",
         "FoG-STAR": "results_mx2/MX2_per_window_FoG-STAR_seed42.csv.gz",
         "tdcsfog (lab)": "results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz",
         "defog (home)": "results_mx2_external/EV_per_window_defog_seed42.csv.gz"}


def analyse(name, W, rng):
    y2 = W["y2"].to_numpy(); s = W[SCORE].to_numpy()
    ev = m1.Ev(W, y2, np.arange(len(W)))
    neg = ev.outside & ~ev.fog
    pre = np.zeros(ev.n, bool)
    for es, a, pe in zip(ev.es, ev.ea, ev.pe):
        lo = a if pe < 0 else max(a, pe + m1.POST_W)
        pre[max(lo, es - m1.PRE_W):max(lo, es - 1)] = True
    pre &= ~ev.fog
    sid = ev.sid; subj = ev.subj
    pats = np.unique(subj); pi = {p: i for i, p in enumerate(pats)}
    C = np.zeros(len(pats)); PN = np.zeros(len(pats))
    for g in np.unique(sid[pre]):
        m = sid == g
        sp = np.sort(s[m & pre]); sn = np.sort(s[m & neg])
        if len(sp) == 0 or len(sn) == 0:
            continue
        lo = np.searchsorted(sn, sp, "left"); hi = np.searchsorted(sn, sp, "right")
        i = pi[subj[m][0]]
        C[i] += lo.sum() + 0.5 * (hi - lo).sum(); PN[i] += len(sp) * len(sn)
    w = np.stack([np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats)) for _ in range(N_BOOT)]).astype(float)
    ok = (w @ PN) > 0
    boot = (w[ok] @ C) / (w[ok] @ PN)
    auc_row = {"cohort": name, "within_segment_AUC_lead1": C.sum() / PN.sum(), "CI_low": np.percentile(boot, 2.5),
               "CI_high": np.percentile(boot, 97.5), "segments_contributing": int(np.unique(sid[pre & np.isin(sid, sid[neg])]).size),
               "patients_contributing": int((PN > 0).sum())}
    base = pd.Series(s[neg]).groupby(sid[neg]).median()
    prof = {o: [] for o in OFFSETS}; psub = {o: [] for o in OFFSETS}
    for es, a, ee, pe in zip(ev.es, ev.ea, ev.ee, ev.pe):
        g = sid[es]
        if g not in base.index:
            continue
        lo = a if pe < 0 else max(a, pe + m1.POST_W)
        for o in OFFSETS:
            j = es + o
            if lo <= j < ev.n and sid[j] == g and (o < 0 or j < ee + m1.POST_W):
                if o < 0 and ev.fog[j]:
                    continue
                prof[o].append(s[j] - base[g]); psub[o].append(subj[j])
    rows = []
    for o in OFFSETS:
        v = np.array(prof[o]); g = np.array(psub[o])
        if len(v) == 0:
            continue
        ps = np.unique(g); sums = np.array([v[g == p].sum() for p in ps]); cnt = np.array([(g == p).sum() for p in ps])
        b = rng.integers(0, len(ps), (N_BOOT, len(ps)))
        bm = sums[b].sum(1) / cnt[b].sum(1)
        rows.append({"cohort": name, "offset_windows": o, "offset_s": o * m1.WIN_S, "n": len(v),
                     "mean_score_minus_segment_baseline": v.mean(), "CI_low": np.percentile(bm, 2.5), "CI_high": np.percentile(bm, 97.5)})
    return auc_row, rows


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    rng = np.random.default_rng(42)
    A, P = [], []
    for name, f in FILES.items():
        if not os.path.exists(f):
            print(f"{name}: {f} not found - skipped", flush=True); continue
        a, p = analyse(name, pd.read_csv(f), rng)
        A.append(a); P += p
        print(f"{name}: done", flush=True)
    A = pd.DataFrame(A).round(3); P = pd.DataFrame(P).round(4)
    A.to_csv(os.path.join(OUT, "PROF_T1_within_segment_auc.csv"), index=False)
    P.to_csv(os.path.join(OUT, "PROF_T2_onset_locked_profile.csv"), index=False)
    print("\n", A.to_string(index=False))
    print("\n", P.pivot(index="offset_s", columns="cohort", values="mean_score_minus_segment_baseline").round(3).to_string())
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
