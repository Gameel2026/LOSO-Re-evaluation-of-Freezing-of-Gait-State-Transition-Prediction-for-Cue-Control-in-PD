import os
import pickle
import numpy as np
import pandas as pd

CACHES = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
          "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")],
          "FoG-STAR": ["results_fogstar_causal"]}
FOGSTAR_CSV = r"D:\FoG-STAR\sensor_data.csv"      
W = 0.5                                             


def load_meta(cands):
    for d in cands:
        f = os.path.join(d, "cache_Shank.pkl")
        if os.path.exists(f):
            return pickle.load(open(f, "rb"))[2]
    return None


def episodes(meta):
    rows = []
    for (s, r, g), m in meta.groupby(["subject", "run", "seg"], sort=False):
        y = m["y2"].to_numpy(); pos = m["pos"].to_numpy(); n = len(y)
        prev_end = None
        for i in range(n):
            if y[i] == 2 and (i == 0 or y[i - 1] != 2):
                j = i
                while j < n and y[j] == 2: j += 1
                rows.append({"subject": s, "run": r, "history_s": pos[i] * W,
                             "gap_prev_s": (pos[i] - prev_end) * W if prev_end is not None else np.nan,
                             "duration_s": (j - i) * W, "stretch_s": n * W})
                prev_end = pos[j - 1] + 1
    return pd.DataFrame(rows)


def main():
    out, summ = [], []
    for ds, cands in CACHES.items():
        meta = load_meta(cands)
        if meta is None:
            print(f"{ds}: cache not found - skipped"); continue
        e = episodes(meta); e.insert(0, "dataset", ds); out.append(e)
        full = (e.history_s >= 4) & ~(e.gap_prev_s < 4)                 
        summ.append({"dataset": ds, "episodes": len(e),
                     "onset in first 0.5 s of a stretch (%)": 100 * (e.history_s == 0).mean(),
                     "history < 4 s (%)": 100 * (e.history_s < 4).mean(),
                     "preceded by FoG within 4 s (%)": 100 * (e.gap_prev_s < 4).mean(),
                     "complete 4 s Pre-FoG (%)": 100 * full.mean(),
                     "median history (s)": e.history_s.median(),
                     "median stretch length (s)": meta.groupby(["subject", "run", "seg"]).size().median() * W})
    pd.concat(out).round(2).to_csv("fogstar_check.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    print("\nRecording available before each FoG onset (descriptive, no model outputs)\n")
    print(pd.DataFrame(summ).round(1).to_string(index=False))
    if not os.path.exists(FOGSTAR_CSV):
        print(f"\nFoG-STAR activity analysis skipped: file not found at {FOGSTAR_CSV}")
    else:
        df = pd.read_csv(FOGSTAR_CSV, usecols=["timestamp", "fog", "activity", "subjectID", "sessionID", "taskID"])
        df = df.sort_values(["subjectID", "sessionID", "taskID", "timestamp"], kind="stable").reset_index(drop=True)
        f = (pd.to_numeric(df["fog"], errors="coerce").fillna(0) > 0).to_numpy()
        act = df["activity"].astype(str).str.strip().str.lower().to_numpy()
        key = df[["subjectID", "sessionID", "taskID"]].astype(str).agg("|".join, axis=1).to_numpy()
        same = np.r_[False, key[1:] == key[:-1]]
        onsets = np.flatnonzero(f & ~(np.r_[False, f[:-1]] & same))
        FS, PRE = 60, 4 * 60                                        
        at_onset, before = [], []
        for i in onsets:
            j0 = i
            while j0 > 0 and key[j0 - 1] == key[i] and i - j0 < PRE: j0 -= 1   
            seg = act[j0:i][~f[j0:i]]                                            
            at_onset.append(act[i])
            before.append(pd.Series(seg).value_counts(normalize=True) if len(seg) else pd.Series(dtype=float))
        print(f"\nFoG-STAR: {len(onsets)} onsets found")
        print("\nActivity annotated at FoG onset (share of episodes, %)\n",
              (100 * pd.Series(at_onset).value_counts(normalize=True)).round(1).to_string())
        B = pd.DataFrame(before).fillna(0)
        print("\nActivity during the 4 s before onset (mean share of time across episodes, %)\n",
              (100 * B.mean()).sort_values(ascending=False).round(1).to_string())
        walk = [c for c in B.columns if "walk" in c]
        if walk:
            w = B[walk].sum(axis=1)
            print(f"\nEpisodes whose 4 s before onset were mostly walking (>50%): {100 * (w > 0.5).mean():.1f}%")
            print(f"Episodes with no walking in the 4 s before onset: {100 * (w == 0).mean():.1f}%")
        print("\nTask at FoG onset (episodes)\n", df.loc[onsets, "taskID"].value_counts().to_string())
    print("\nDone. Per-episode values in fogstar_check.csv")


if __name__ == "__main__":
    main()
