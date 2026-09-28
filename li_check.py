import os
import glob
import numpy as np
import pandas as pd
from li_io import read_task, subject_from_folder, LI_IO_VERSION
print("li_io version:", LI_IO_VERSION)   

from paths import LI_RAW_ROOT as LI_ROOT
from paths import LI_CONVERTED as OUT_ROOT  
FS = 500
SENSORS = {"LShank": 31, "RShank": 38, "Waist": 45, "Arm": 52}   


rows = []
files = sorted(glob.glob(os.path.join(LI_ROOT, "**", "task_*.txt"), recursive=True))
if not files:
    raise FileNotFoundError(f"No task_*.txt found under {LI_ROOT} - check LI_ROOT")
for path in files:
    a = read_task(path)
    fo = os.path.relpath(os.path.dirname(path), LI_ROOT)
    row = {"folder": fo, "subject": subject_from_folder(fo), "task": os.path.basename(path),
           "n_cols": a.shape[1], "duration_s": round(len(a) / FS, 1), "FoG_%": round(100 * np.mean(a[:, -1] > 0.5), 1)}
    for name, c in SENSORS.items():
        row[f"{name}_zero_%"] = round(100 * np.mean(np.all(np.abs(a[:, c:c + 3]) < 1e-12, axis=1)), 1)
    rows.append(row)
    print(f"read {row['folder']}/{row['task']}")

df = pd.DataFrame(rows)
pd.set_option("display.max_rows", None); pd.set_option("display.width", 250)
print("\n", df.to_string(index=False))
print("\nPer folder (mean over tasks):")
print(df.groupby(["subject", "folder"])[[c for c in df.columns if c.endswith("_%")]].mean().round(1).to_string())

conv = sorted(glob.glob(os.path.join(OUT_ROOT, "S*R*.txt")))
if conv:
    print("\nConverted files (label 0 = removed):")
    for p in conv:
        lab = np.loadtxt(p, usecols=10)
        print(f"  {os.path.basename(p)}: kept {100 * np.mean(lab > 0):.0f}%  FoG {100 * np.mean(lab == 2):.0f}%")
df.to_csv("li_check.csv", index=False)
print("\nSaved li_check.csv")
