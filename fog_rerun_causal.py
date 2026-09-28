import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from fog_pipeline import (Config, HYBRID_23, extract_features, load_daphnet, make_rf,
                          rank_by_mi, relabel_four_class, segment_windows, smote,
                          weighted_scores, per_class_metrics)
from fog_study import (binary_cue_loso, binary_window_metrics, event_metrics,
                       loso_decoders)
from fog_stats import per_subject, run_tests

from paths import DAPHNET_ROOT as ROOT
OUT = "results_causal"
QUICK = False                 
RUN = {"E1": True, "E23": True, "E45": True, "STATS": True}
SENSORS = {"Shank": ("ank",), "Thigh": ("thi",), "Trunk": ("tr",),
           "All three": ("ank", "thi", "tr")}
LABELS = ["No-FoG", "Pre-FoG", "FoG", "Post-FoG"]
SENSOR_ROOTS = {}   


def cfg_for(sensors):
    return Config(causal_filter=True, sensors=sensors, n_estimators=30 if QUICK else 100)


def data_for(raw, name):
    path = f"{OUT}/cache_{name.replace(' ', '_')}.pkl"
    cfg = cfg_for(SENSORS[name])                 
    if name in SENSOR_ROOTS and not os.path.exists(path):
        raw = load_daphnet(SENSOR_ROOTS[name])
    if os.path.exists(path):
        _, X, meta, y4 = pickle.load(open(path, "rb"))
        return cfg, X, meta, y4
    windows, meta = segment_windows(raw, cfg)
    X = extract_features(windows, cfg)
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)
    pickle.dump((cfg, X, meta, y4), open(path, "wb"))
    return cfg, X, meta, y4


def save_cm(y, p, name):
    cm = confusion_matrix(y, p, labels=[0, 1, 2, 3])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(f"{OUT}/CM_{name}.csv")


def summary(y, p):
    pc = per_class_metrics(y, p)
    return {**weighted_scores(y, p),
            "PreFoG_recall": pc.loc["Pre-FoG", "recall"],
            "PostFoG_recall": pc.loc["Post-FoG", "recall"],
            "transition_recall": pc.loc[["Pre-FoG", "Post-FoG"], "recall"].mean()}


def window_cv(X, y4, cfg, smote_before):
    Xf = X[HYBRID_23].to_numpy()
    if smote_before:                                    
        Xf, y = smote(Xf, y4, cfg.smote_k, cfg.seed)
    else:
        y = y4
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed).split(Xf, y):
        Xtr, ytr = (Xf[tr], y[tr]) if smote_before else smote(Xf[tr], y[tr], cfg.smote_k, cfg.seed)
        pred[te] = make_rf(cfg).fit(Xtr, ytr).predict(Xf[te])
    return y, pred


def loso_fixed_features(X, y4, meta, cfg):
    groups = meta["subject"].to_numpy()
    pred = np.empty_like(y4)
    Xf = X[HYBRID_23].to_numpy()
    for s in np.unique(groups):
        tr, te = groups != s, groups == s
        Xb, yb = smote(Xf[tr], y4[tr], cfg.smote_k, cfg.seed)
        pred[te] = make_rf(cfg).fit(Xb, yb).predict(Xf[te])
        print(f"  P2 fold subject {s:02d} done")
    return pred


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load_daphnet(SENSOR_ROOTS.get("Shank", ROOT))
    cfg, X, meta, y4 = data_for(raw, "Shank")
    y2 = meta["y2"].to_numpy()
    print(f"windows: {len(X)}  FoG windows: {(y2 == 2).sum()}")

    preds = None
    if RUN["E1"] or RUN["E23"] or RUN["STATS"]:
        print("LOSO decoders (P3 + Exp. 2) ...")
        preds = loso_decoders(X, y4, meta, cfg)

    if RUN["E1"]:
        rows = {}
        print("P0 original protocol ...")
        y, p = window_cv(X, y4, cfg, smote_before=True);  rows["P0 Original"] = summary(y, p);  save_cm(y, p, "P0")
        print("P1 SMOTE inside folds ...")
        y, p = window_cv(X, y4, cfg, smote_before=False); rows["P1 SMOTE in folds"] = summary(y, p); save_cm(y, p, "P1")
        print("P2 LOSO, published features ...")
        p = loso_fixed_features(X, y4, meta, cfg);          rows["P2 LOSO fixed features"] = summary(y4, p); save_cm(y4, p, "P2")
        p = preds["RF (per-window)"];                        rows["P3 LOSO in-fold features"] = summary(y4, p); save_cm(y4, p, "P3")
        e1 = pd.DataFrame(rows).T.round(4); e1.to_csv(f"{OUT}/E1_protocols.csv"); print("\nE1\n", e1)

    if RUN["E23"]:
        e2 = pd.DataFrame({k: summary(y4, p) for k, p in preds.items()}).T.round(4)
        e2.to_csv(f"{OUT}/E2_decoders.csv"); print("\nE2\n", e2)
        save_cm(y4, preds["HMM forward (causal)"], "hmm")
        e3 = pd.DataFrame({k: {**binary_window_metrics(y4, p), **event_metrics(y2, p, meta, cfg)}
                           for k, p in preds.items()}).T.round(3)
        e3.to_csv(f"{OUT}/E3_event_level.csv"); print("\nE3\n", e3)

    tables = []
    if preds is not None:
        for k in ["RF (per-window)", "HMM forward (causal)"]:
            tables.append(per_subject(k, preds[k], y4, meta, cfg))

    if RUN["E45"] or RUN["STATS"]:
        rows = {}
        for name in SENSORS:
            print(f"\nE4/E5: {name} ...")
            c, Xs, ms, ys = data_for(raw, name)
            pred, _, thr = binary_cue_loso(Xs, ys, ms, c)
            thr.to_csv(f"{OUT}/E4_thresholds_{name.replace(' ', '_')}.csv")
            rows[name] = {"n_features": Xs.shape[1], **binary_window_metrics(ys, pred),
                          **event_metrics(ms["y2"].to_numpy(), pred, ms, c)}
            pd.DataFrame(rows).T.round(3).to_csv(f"{OUT}/E4_E5_sensors.csv")
            if name in ("Shank", "Trunk"):
                tables.append(per_subject(f"E4 {name}", pred, ys, ms, c))
        print("\nE4/E5\n", pd.DataFrame(rows).T.round(3))

    if RUN["STATS"] and tables:
        table = pd.concat(tables, ignore_index=True)
        table.round(3).to_csv(f"{OUT}/S1_per_subject.csv", index=False)
        tests = run_tests(table); tests.round(4).to_csv(f"{OUT}/S2_wilcoxon.csv", index=False)
        pd.set_option("display.max_columns", None); pd.set_option("display.width", 250)
        print("\nWilcoxon\n", tests.round(4))
    print(f"\nDone. All results in {OUT}/")


if __name__ == "__main__":
    main()
