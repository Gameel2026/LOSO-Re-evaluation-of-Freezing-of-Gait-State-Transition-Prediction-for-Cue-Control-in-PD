import os
import re
import glob
import numpy as np
from scipy.signal import resample_poly
from li_io import read_task, subject_from_folder, EXPECTED_COLS, LI_IO_VERSION
print("li_io version:", LI_IO_VERSION)   # must print v3-end-anchored

from paths import LI_RAW_ROOT as LI_ROOT
from paths import LI_CONVERTED as OUT_ROOT   # <OUT_ROOT>_shank, _trunk, _shankwaist are written
# <OUT_ROOT>_shank (left shank valid), <OUT_ROOT>_trunk (waist valid), <OUT_ROOT>_all (all three valid)
REQUIRE = {"shank": ("ank",), "trunk": ("tr",), "shankwaist": ("ank", "tr")}
# "ank" = LEFT shank when it is recorded, otherwise the RIGHT shank (chosen per task file)
DRY_RUN = True
FS_IN, FS_OUT = 500, 64
UP, DOWN = 16, 125                       
# If several folders belong to the same patient, map folder name -> patient number here,
# e.g. {"001": 1, "002": 1, "003": 2}. Empty = every folder is a different patient.
SUBJECT_MAP = {}   # normally empty: the patient number is taken from the top-level folder name

IMU0 = 31                                
def acc_cols(sensor):                    
    s = IMU0 + 7 * sensor
    return [s, s + 1, s + 2]


def pick_shank(arr):
    zero = lambda a: np.mean(np.all(np.abs(a) < 1e-12, axis=1))
    left, right = arr[:, acc_cols(0)], arr[:, acc_cols(1)]
    return (left, "L") if zero(left) < 0.5 or zero(right) >= 0.5 else (right, "R")


def convert_one(arr, require):
    lab_in = arr[:, -1]
    shank, side = pick_shank(arr)
    other = arr[:, acc_cols(1)] if side == "L" else arr[:, acc_cols(0)]
    acc = {"ank": shank, "thi": other, "tr": arr[:, acc_cols(2)]}
    rs = {k: resample_poly(v, UP, DOWN, axis=0) for k, v in acc.items()}
    n = min(len(v) for v in rs.values())
    src = np.minimum(np.round(np.arange(n) * FS_IN / FS_OUT).astype(int), len(arr) - 1)
    label = np.where(lab_in[src] > 0.5, 2, 1)
    zero = lambda a: np.all(np.abs(a) < 1e-12, axis=1)
    invalid = np.zeros(n, bool)
    for k in require:                                   
        invalid |= zero(acc[k][src])
    label[invalid] = 0
    t = np.arange(n) * 1000.0 / FS_OUT
    out = np.column_stack([t, rs["ank"][:n], rs["thi"][:n], rs["tr"][:n], label])
    return out, invalid.mean(), (label == 2).mean(), side


def main():
    files = sorted(glob.glob(os.path.join(LI_ROOT, "**", "task_*.txt"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No task_*.txt found under {LI_ROOT}")
    folders = sorted({os.path.relpath(os.path.dirname(f), LI_ROOT) for f in files})
    print(f"Found {len(files)} task files in {len(folders)} folders:")
    subj_of = {}
    for fo in folders:
        subj_of[fo] = subject_from_folder(fo, SUBJECT_MAP)
        n = sum(os.path.dirname(os.path.relpath(f, LI_ROOT)) == fo for f in files)
        print(f"  folder '{fo}' -> subject {subj_of[fo]:02d}  ({n} tasks)")
    arr = read_task(files[0])
    print(f"\nFirst file: {files[0]}\n  shape {arr.shape} (expected {EXPECTED_COLS} columns), "
          f"duration {arr.shape[0] / FS_IN:.1f} s, FoG fraction {np.mean(arr[:, -1] > 0.5):.2f}")
    if DRY_RUN:
        print("\nDRY RUN only. Check the subject mapping above, then set DRY_RUN = False.")
        return
    for tag in REQUIRE:
        os.makedirs(f"{OUT_ROOT}_{tag}", exist_ok=True)
    run_counter = {}
    for f in files:
        fo = os.path.relpath(os.path.dirname(f), LI_ROOT); s = subj_of[fo]
        run_counter[s] = run_counter.get(s, 0) + 1; r = run_counter[s]
        arr = read_task(f)
        if arr.shape[1] != EXPECTED_COLS:
            print(f"  WARNING {f}: {arr.shape[1]} columns (expected {EXPECTED_COLS}) - skipped"); continue
        msg = []; fog = float(np.mean(arr[:, -1] > 0.5)); side = pick_shank(arr)[1]
        for tag, req in REQUIRE.items():
            out, inv, _, _ = convert_one(arr, req)
            if inv < 1.0:                                   
                np.savetxt(os.path.join(f"{OUT_ROOT}_{tag}", f"S{s:02d}R{r:02d}.txt"), out, fmt="%.6g")
            msg.append(f"{tag}: {'kept' if inv < 1.0 else 'EXCLUDED'}")
        print(f"  {fo}/{os.path.basename(f)} -> S{s:02d}R{r:02d}  ({len(arr) / FS_IN:.0f} s, FoG {fog:.0%}, shank {side}; " + ", ".join(msg) + ")")
    print(f"\nDone. DAPHNET-format files in {OUT_ROOT}_shank / _trunk / _shankwaist")


if __name__ == "__main__":
    main()
