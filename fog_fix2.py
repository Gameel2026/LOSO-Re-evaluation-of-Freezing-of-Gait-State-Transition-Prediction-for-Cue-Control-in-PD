import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

from fog_pipeline import (Config, HYBRID_23, NOFOG, FOG, contiguous_runs, make_rf,
                          smote, weighted_scores, per_class_metrics)
from fog_study import binary_cue_loso, event_metrics, apply_hysteresis, sequences

OUT = "results_causal"
QUICK = False
N_SHIFTS = 200


def load():
    _, X, meta, y4 = pickle.load(open(f"{OUT}/cache_Shank.pkl", "rb"))
    cfg = Config(causal_filter=True, n_estimators=30 if QUICK else 100)
    return cfg, X, meta, y4


def p0_real_only(X, y4, cfg):
    Xf = X[HYBRID_23].to_numpy()
    Xb, yb = smote(Xf, y4, cfg.smote_k, cfg.seed)       # real samples come first
    is_real = np.zeros(len(yb), bool); is_real[:len(y4)] = True
    pred = np.empty_like(yb)
    for tr, te in StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed).split(Xb, yb):
        pred[te] = make_rf(cfg).fit(Xb[tr], yb[tr]).predict(Xb[te])
    rows = {}
    for name, m in [("P0 all test windows", np.ones(len(yb), bool)),
                    ("P0 real test windows only", is_real)]:
        pc = per_class_metrics(yb[m], pred[m])
        rows[name] = {**weighted_scores(yb[m], pred[m]),
                      "PreFoG_recall": pc.loc["Pre-FoG", "recall"],
                      "PostFoG_recall": pc.loc["Post-FoG", "recall"],
                      "transition_recall": pc.loc[["Pre-FoG", "Post-FoG"], "recall"].mean(),
                      "n_test_windows": int(m.sum())}
    return pd.DataFrame(rows).T


def timely_activation(y2, pred, meta, cfg):
    n_ep = n_any = n_timely = 0
    for _, idx in sequences(meta):
        on = pred[idx] == FOG
        runs = contiguous_runs(on)
        for s, e in contiguous_runs(y2[idx] == 2):
            n_ep += 1
            if s > 0 and on[s - 1]:
                n_any += 1
                start = next(a for a, b in runs if a < s <= b)
                if s - start <= cfg.pre_w:
                    n_timely += 1
    return {"episodes": n_ep, "pre_onset_%": 100 * n_any / n_ep,
            "timely_pre_onset_%": 100 * n_timely / n_ep}


def circular_shift(pred, meta, rng):
    out = pred.copy()
    for _, idx in sequences(meta):
        if len(idx) > 1:
            out[idx] = np.roll(pred[idx], rng.integers(1, len(idx)))
    return out


def main():
    cfg, X, meta, y4 = load()
    y2 = meta["y2"].to_numpy()

    print("A) P0 on real test windows ...")
    a = p0_real_only(X, y4, cfg).round(4)
    a.to_csv(f"{OUT}/F2_A_P0_real_only.csv"); print(a)

    print("\nB) Proposed controller (shank) ...")
    pred, score, thr = binary_cue_loso(X, y4, meta, cfg)
    ev = event_metrics(y2, pred, meta, cfg)
    tim = timely_activation(y2, pred, meta, cfg)

    rng = np.random.default_rng(cfg.seed)
    null = []
    for i in range(N_SHIFTS):
        sp = circular_shift(pred, meta, rng)
        e = event_metrics(y2, sp, meta, cfg)
        t = timely_activation(y2, sp, meta, cfg)
        null.append({"pre_onset_%": e["predicted_before_onset_%"], "timely_pre_onset_%": t["timely_pre_onset_%"],
                     "detected_%": e["detected_%"]})
    null = pd.DataFrame(null)
    obs = {"pre_onset_%": ev["predicted_before_onset_%"], "timely_pre_onset_%": tim["timely_pre_onset_%"],
           "detected_%": ev["detected_%"]}
    b = pd.DataFrame({k: {"observed": obs[k], "surrogate_mean": null[k].mean(),
                          "surrogate_95th": null[k].quantile(0.95),
                          "p_one_sided": (1 + (null[k] >= obs[k]).sum()) / (1 + N_SHIFTS)} for k in obs}).T.round(4)
    b.to_csv(f"{OUT}/F2_B_surrogate.csv"); print(b)

    rows = []
    allmask = np.ones(len(y4), bool)
    for on_t in np.arange(0.30, 0.91, 0.05):
        for off_t in np.arange(0.10, on_t + 1e-9, 0.10):
            p = np.where(apply_hysteresis(score, meta, allmask, on_t, off_t), FOG, NOFOG)
            e = event_metrics(y2, p, meta, cfg); t = timely_activation(y2, p, meta, cfg)
            rows.append({"theta_on": round(on_t, 2), "theta_off": round(off_t, 2),
                         "pre_onset_%": e["predicted_before_onset_%"], "timely_pre_onset_%": t["timely_pre_onset_%"],
                         "detected_%": e["detected_%"], "false_alarms_per_hour": e["false_alarms_per_hour"],
                         "median_cue_off_delay_s": e["median_cue_off_delay_s"]})
    sw = pd.DataFrame(rows).round(3); sw.to_csv(f"{OUT}/F2_B3_threshold_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    sc = ax.scatter(sw["false_alarms_per_hour"], sw["timely_pre_onset_%"], c=sw["theta_off"], cmap="viridis", s=18)
    ax.set_xscale("log"); ax.set_xlabel("False alarms per hour (log scale)")
    ax.set_ylabel("Timely pre-onset activation (%)"); plt.colorbar(sc, label="theta_off")
    ax.scatter([ev["false_alarms_per_hour"]], [tim["timely_pre_onset_%"]], marker="*", s=200, c="red",
               edgecolors="black", label="Selected (tuned on training)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(f"{OUT}/F2_threshold_sweep.png", dpi=300); plt.show()

    summ = pd.DataFrame([{**ev, **tim}]).round(3); summ.to_csv(f"{OUT}/F2_B_controller_summary.csv", index=False)
    print("\nController:", summ.to_dict("records")[0])
    print(f"\nDone. Files F2_* in {OUT}/")


if __name__ == "__main__":
    main()
