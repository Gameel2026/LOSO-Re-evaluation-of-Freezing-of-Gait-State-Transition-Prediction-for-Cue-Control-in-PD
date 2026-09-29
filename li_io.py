import re
import numpy as np
import pandas as pd

EXPECTED_COLS = 60
LI_IO_VERSION = "v3-end-anchored"


def read_task(path):
    with open(path) as f:
        first = f.readline(); second = f.readline() or first
    sep = "," if "," in second else ("\t" if "\t" in second else r"\s+")
    tok = [t for t in re.split(r"[,\t ]+", first.strip()) if t]
    numeric_first_line = sum(bool(re.fullmatch(r"[\d.+\-eE:]+", t)) for t in tok) >= 0.8 * max(len(tok), 1)
    df = pd.read_csv(path, sep=sep, header=None if numeric_first_line else 0, engine="c")
    data = df.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")                
    if data.shape[1] < EXPECTED_COLS - 1:
        raise ValueError(f"{path}: only {data.shape[1]} numeric columns found")
    core = data.iloc[:, -(EXPECTED_COLS - 1):]          
    ok = core.iloc[:, -29:].notna().all(axis=1)         
    core = core[ok]
    return np.column_stack([np.zeros(len(core)), core.to_numpy(float)])


def subject_from_folder(rel_folder, subject_map=None):
    subject_map = subject_map or {}
    if rel_folder in subject_map:
        return subject_map[rel_folder]
    top = re.split(r"[\\/]", rel_folder)[0]
    if top in subject_map:
        return subject_map[top]
    m = re.search(r"\d+", top)
    if not m:
        raise ValueError(f"Cannot infer subject number from folder '{rel_folder}'; add it to SUBJECT_MAP")
    return int(m.group())
