import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import signal

import fog_improve as fi
import fog_floor_sensitivity as fs
from fog_pipeline import Config, load_daphnet, segment_windows, extract_features, relabel_four_class
from fog_study import event_metrics
from fog_pipeline import FOG, NOFOG
from fog_stats import holm

OUT = "results_hiss_sensitivity"
RUN = {"R1": True, "R2": False, "R3": True}
N_SHIFTS = 2000
FOGSTAR_CSV = r"D:\New research fog work\FoG STAR\sensor_data.csv"
FOGSTAR_OUT_CAUSAL = r"D:\FoG-STAR\daphnet_format_causalrs"      
ORIGINAL = {"DAPHNET": ["results_causal", os.path.join("New Results", "results_causal")],
            "Multimodal FoG": ["results_li_causal", os.path.join("New Results", "results_li_causal")],
            "FoG-STAR": ["results_fogstar_causal", os.path.join("New Results", "results_fogstar_causal")]}
CAUSAL_SRC = {"Multimodal FoG (causal resampling)": "results_hiss_causalrs_li",
              "FoG-STAR (causal resampling)": "results_hiss_causalrs_fogstar"}
SHIFT_WINDOWS = [-2, -1, 1, 2]                                      
FLOORS = [None, 0.70]
fl = lambda f: "no floor" if f is None else "floor 0.70"
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")


def causal_resample(x, up, down, axis=0):
    x = np.asarray(x, float)
    n_in = x.shape[0]
    n_out = int(np.ceil(n_in * up / down))
    idx = np.minimum(np.floor(np.arange(n_out) * down / up).astype(int), n_in - 1)   
    if up < down:                                                    
        sos = signal.butter(8, 0.9 * up / down, btype="low", output="sos")
        zi = signal.sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
        x, _ = signal.sosfilt(sos, x, axis=0, zi=zi)
    return x[idx]                                                    


def convert_causal():
    import li_convert as lc
    import fogstar_convert as fc
    from paths import LI_CONVERTED
    lc.resample_poly = causal_resample; lc.DRY_RUN = False; lc.OUT_ROOT = LI_CONVERTED + "_causalrs"
    fc.resample_poly = causal_resample; fc.DRY_RUN = False; fc.CSV = FOGSTAR_CSV; fc.FOGSTAR_OUT = FOGSTAR_OUT_CAUSAL
    if not os.path.isdir(lc.OUT_ROOT + "_shank"):
        print("Converting Multimodal FoG with causal resampling ..."); lc.main()
    if not os.path.isdir(FOGSTAR_OUT_CAUSAL + "_ankle"):
        print("Converting FoG-STAR with causal resampling ..."); fc.main()
    return {"Multimodal FoG (causal resampling)": lc.OUT_ROOT + "_shank",
            "FoG-STAR (causal resampling)": FOGSTAR_OUT_CAUSAL + "_ankle"}


def build_cache(folder, src):
    path = os.path.join(src, "cache_Shank.pkl")
    if os.path.exists(path):
        return
    os.makedirs(src, exist_ok=True)
    cfg = Config(causal_filter=True, sensors=("ank",))
    windows, meta = segment_windows(load_daphnet(folder), cfg)
    X = extract_features(windows, cfg)
    pickle.dump((cfg, X, meta, relabel_four_class(meta, cfg.pre_w, cfg.post_w)), open(path, "wb"))
    print(f"  cache written: {path} ({len(X)} windows)")


def run_controller(name, candidates):
    fs.DATASETS = {name: candidates if isinstance(candidates, list) else [candidates]}
    fs.FLOORS = FLOORS
    return fs.run_dataset(name, fi.cfg_())                           


def surrogate_rows(name, y4, meta, on, cfg):
    rows = []
    for f in FLOORS:
        s = fi.summarise(y4, on[f], meta, cfg)
        sg = fi.surrogate(y4, on[f], meta, cfg)
        for k in ["timely_pre_onset_%", "detected_%"]:
            rows.append({"dataset": name, "controller": fl(f), "metric": k, **sg.loc[k].to_dict(),
                         "cue_specificity": s["cue_specificity"], "false_alarms_per_hour": s["false_alarms_per_hour"]})
    return rows


def shift_episodes(meta, k):
    y2 = meta["y2"].to_numpy().copy(); new = np.ones_like(y2)
    from fog_study import sequences
    for _, idx in sequences(meta):
        f = (y2[idx] == 2); g = np.zeros_like(f)
        if k > 0: g[k:] = f[:-k]
        elif k < 0: g[:k] = f[-k:]
        else: g = f
        new[idx] = np.where(g, 2, 1)
    m = meta.copy(); m["y2"] = new
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
    fi.N_SHIFTS = N_SHIFTS
    cfg = fi.cfg_()
    original = {}

    if RUN["R2"] or RUN["R3"] or RUN["R1"]:
        for ds, cand in ORIGINAL.items():
            if ds == "DAPHNET" and not RUN["R2"]:
                continue
            print(f"\n=== Controller on {ds} (original processing) ===")
            original[ds] = run_controller(ds, cand)

    orig_rows = {}
    def orig(ds):
        if ds not in orig_rows:
            y4, meta, on, _ = original[ds]; orig_rows[ds] = surrogate_rows(ds, y4, meta, on, cfg)
        return orig_rows[ds]

    if RUN["R2"]:
        rows = []
        for ds in original:
            rows += orig(ds)
        R2 = pd.DataFrame(rows); R2["p_holm_family"] = holm(R2["p_one_sided"].to_numpy())
        R2.round(4).to_csv(f"{OUT}/R2_surrogate_2000_holm.csv", index=False)
        print(f"\nR2 surrogate with {N_SHIFTS} shifts, Holm across all {len(R2)} tests\n", R2.round(4).to_string(index=False))

    if RUN["R1"]:
        folders = convert_causal()
        rows = []
        for name, folder in folders.items():
            build_cache(folder, CAUSAL_SRC[name])
            print(f"\n=== Controller on {name} ===")
            y4, meta, on, _ = run_controller(name, CAUSAL_SRC[name])
            rows += surrogate_rows(name, y4, meta, on, cfg)
            base = "Multimodal FoG" if name.startswith("Multimodal") else "FoG-STAR"
            rows += [dict(r, dataset=f"{base} (original processing)") for r in orig(base)]
        R1 = pd.DataFrame(rows).drop_duplicates(subset=["dataset", "controller", "metric"])
        R1.round(4).to_csv(f"{OUT}/R1_causal_resampling.csv", index=False)
        print("\nR1 causal resampling vs original processing\n", R1.round(3).to_string(index=False))

    if RUN["R3"]:
        y4, meta, on, _ = original["FoG-STAR"]; rows = []
        for k in [0] + SHIFT_WINDOWS:
            m = shift_episodes(meta, k)
            for f in FLOORS:
                sg = fi.surrogate(y4, on[f], m, cfg)
                for met in ["timely_pre_onset_%", "detected_%"]:
                    rows.append({"onset_shift_s": 0.5 * k, "controller": fl(f), "metric": met, **sg.loc[met].to_dict()})
            print(f"  shift {0.5 * k:+.1f} s done")
        R3 = pd.DataFrame(rows); R3.round(4).to_csv(f"{OUT}/R3_onset_shift.csv", index=False)
        print("\nR3 FoG-STAR onset shift (controller output unchanged)\n", R3.round(3).to_string(index=False))

    print(f"\nDone. Files in {OUT}/")


if __name__ == "__main__":
    main()