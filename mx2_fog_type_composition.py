import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kaggle_convert as kc

OUT = "results_fog_type_composition"
MAPPING = os.path.join(kc.OUT, "kaggle_subject_mapping.csv")
COHORTS = {"tdcsfog_back": "tdcsfog (lab)", "defog_back": "defog (home)"}
TYPES = ["StartHesitation", "Turn", "Walking"]
REGISTERED_EPISODES = {"tdcsfog (lab)": 949, "defog (home)": 689}


def type_track(d, kind, n):
    sp = kc.SPEC[kind]
    src = np.minimum(np.round(np.arange(n) * sp["fs"] / kc.FS_OUT).astype(int), len(d) - 1)
    if kind == "notype":
        return np.where(d["Event"].to_numpy()[src] > 0, "Unspecified", "").astype(object)
    v = d[TYPES].to_numpy()[src]
    return np.where(v.sum(1) > 0, np.array(TYPES)[v.argmax(1)], "").astype(object)


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", None)
    M = pd.read_csv(MAPPING)
    rows = []
    for _, r in M.iterrows():
        path = os.path.join(kc.ROOT, "train", r["kind"], f"{r['Id']}.csv")
        res = kc.convert_file(path, r["kind"])
        if res is None:
            continue
        lab = res[1]
        t = type_track(pd.read_csv(path), r["kind"], len(lab))
        for s, e in kc.runs_of(lab == 2):
            seg = t[s:e]; seg = seg[seg != ""]
            vals, cnt = np.unique(seg, return_counts=True)
            rows.append({"cohort": COHORTS[r["folder"]], "subject": r["S"][:3], "kind": r["kind"],
                         "fog_type": vals[cnt.argmax()] if len(vals) else "Unspecified",
                         "mixed_types": int(len(vals) > 1), "duration_s": (e - s) / kc.FS_OUT})
    E = pd.DataFrame(rows)
    E.to_csv(os.path.join(OUT, "FTC_T2_episodes.csv"), index=False)
    T = []
    for c, g in E.groupby("cohort", sort=False):
        for ty, h in g.groupby("fog_type"):
            T.append({"cohort": c, "fog_type": ty, "episodes": len(h), "episodes_%": 100 * len(h) / len(g),
                      "patients": h.subject.nunique(), "fog_time_s": h.duration_s.sum(),
                      "fog_time_%": 100 * h.duration_s.sum() / g.duration_s.sum(),
                      "median_duration_s": h.duration_s.median(), "episodes_with_mixed_types": int(h.mixed_types.sum())})
    T1 = pd.DataFrame(T).round(2)
    T1.to_csv(os.path.join(OUT, "FTC_T1_type_composition.csv"), index=False)
    chk = pd.DataFrame([{"cohort": c, "episodes_counted": int((E.cohort == c).sum()), "episodes_registered": n,
                         "equal": int((E.cohort == c).sum()) == n} for c, n in REGISTERED_EPISODES.items()])
    chk.to_csv(os.path.join(OUT, "FTC_T0_episode_check.csv"), index=False)
    print("\n", chk.to_string(index=False))
    print("\n", T1.to_string(index=False))
    print(f"\nDone. Files in {os.path.join(os.getcwd(), OUT)}")


if __name__ == "__main__":
    main()
