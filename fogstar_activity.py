import os
import glob
import numpy as np
import pandas as pd

FOGSTAR_CSV = r"D:\FoG-STAR\sensor_data.csv"     
FS, PRE_S = 60, 4.0                               


def locate():
    if os.path.exists(FOGSTAR_CSV):
        return FOGSTAR_CSV
    for root in [r"D:\FoG-STAR", r"D:\\", os.getcwd(), os.path.expanduser("~")]:
        hits = glob.glob(os.path.join(root, "**", "sensor_data.csv"), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError("sensor_data.csv not found - set FOGSTAR_CSV at the top of this script")


def main():
    path = locate(); print(f"Using: {path}")
    df = pd.read_csv(path, usecols=["timestamp", "activity", "fog", "subjectID", "sessionID", "taskID"])
    df["activity"] = df["activity"].astype(str).str.strip().str.lower()
    print("Activity labels in the file:", sorted(df["activity"].unique()))
    n_pre = int(PRE_S * FS); rows = []
    for (s, se, t), g in df.groupby(["subjectID", "sessionID", "taskID"], sort=True):
        g = g.sort_values("timestamp", kind="stable")
        fog = (pd.to_numeric(g["fog"], errors="coerce").fillna(0) > 0).to_numpy()
        act = g["activity"].to_numpy()
        onsets = np.flatnonzero(fog & ~np.r_[False, fog[:-1]])
        for i in onsets:
            pre = slice(max(0, i - n_pre), i)
            a = pd.Series(act[pre]); f = fog[pre]
            rows.append({"subject": s, "session": se, "task": t, "activity_at_onset": act[i],
                         "pre_seconds_available": (i - max(0, i - n_pre)) / FS,
                         "pre_fog_%": 100 * f.mean() if len(f) else np.nan,
                         **{f"pre_{k}_%": 100 * v for k, v in a.value_counts(normalize=True).items()},
                         "pre_dominant": a.mode().iloc[0] if len(a) else "none"})
    E = pd.DataFrame(rows).fillna({c: 0 for c in [c for c in pd.DataFrame(rows).columns if c.startswith("pre_") and c.endswith("_%")]})
    E.round(1).to_csv("fogstar_activity.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
    print(f"\nEpisodes: {len(E)}")
    print("\nActivity at FoG onset (% of episodes):\n", (100 * E.activity_at_onset.value_counts(normalize=True)).round(1).to_string())
    print("\nDominant activity in the 4 s before onset (% of episodes):\n", (100 * E.pre_dominant.value_counts(normalize=True)).round(1).to_string())
    act_cols = [c for c in E.columns if c.startswith("pre_") and c.endswith("_%") and c != "pre_fog_%"]
    print("\nMean share of the 4 s before onset spent in each activity (%):\n", E[act_cols].mean().sort_values(ascending=False).round(1).to_string())
    walk = [c for c in act_cols if "walk" in c]
    if walk:
        w = E[walk].sum(axis=1)
        print(f"\nEpisodes with walking during at least half of the 4 s before onset: {100 * (w >= 50).mean():.1f}%")
        print(f"Episodes with no walking at all in the 4 s before onset:           {100 * (w == 0).mean():.1f}%")
    print("\nTask at FoG onset (% of episodes):\n", (100 * E.task.value_counts(normalize=True)).round(1).to_string())
    print("\nDone. Per-episode values in fogstar_activity.csv")


if __name__ == "__main__":
    main()
