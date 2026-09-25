import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from fog_pipeline import Config, extract_features, load_daphnet, relabel_four_class, segment_windows
from fog_study import loso_decoders

ROOT = r"D:\dataset\dataset"
LABELS = ["No-FoG", "Pre-FoG", "FoG", "Post-FoG"]
FILES = {"RF (per-window)": "CM_rf", "Moving avg (k=3)": "CM_moving_avg",
         "HMM forward (causal)": "CM_hmm", "Viterbi (offline)": "CM_viterbi"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8})


def draw(ax, cm, title, show_ylabels=True):
    """Row-normalised confusion matrix with percentages and counts."""
    pct = cm / cm.sum(1, keepdims=True) * 100
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)          
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{pct[i, j]:.1f}%\n({cm[i, j]:,})", ha="center", va="center",
                    fontsize=5.6, color="white" if pct[i, j] > 55 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS, rotation=35, ha="right", rotation_mode="anchor")  # الكلام مايل
    ax.set_yticklabels(LABELS if show_ylabels else [])
    ax.set_xlabel("Predicted class")
    if show_ylabels:
        ax.set_ylabel("True class")
    ax.set_title(title, fontsize=8, loc="left")
    return im


def save_png(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.show()


def main():
    cfg = Config()
    os.makedirs("results", exist_ok=True)
    windows, meta = segment_windows(load_daphnet(ROOT), cfg)
    X = extract_features(windows, cfg)
    y4 = relabel_four_class(meta, cfg.pre_w, cfg.post_w)

    # ---- one confusion matrix per decoder ----
    preds = loso_decoders(X, y4, meta, cfg)
    for name, pred in preds.items():
        cm = confusion_matrix(y4, pred, labels=[0, 1, 2, 3])
        base = f"results/{FILES[name]}"
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(base + ".csv")
        print(f"\n{name}\n{cm}\naccuracy: {np.trace(cm) / cm.sum():.4f}")
        fig, ax = plt.subplots(figsize=(90 / 25.4, 80 / 25.4))
        draw(ax, cm, name)
        save_png(fig, base + ".png")

    # ---- three-panel figure for the manuscript (Fig. 3) ----
    needed = ["results/CM_original.csv", "results/CM_loso.csv", "results/CM_hmm.csv"]
    if all(os.path.exists(f) for f in needed):
        cms = [pd.read_csv(f, index_col=0).to_numpy() for f in needed]
        fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 62 / 25.4))
        for ax, cm, t in zip(axes, cms, ["(a) Original protocol", "(b) LOSO", "(c) LOSO + HMM filtering"]):
            draw(ax, cm, t)
        save_png(fig, "results/Fig3_three_panels.png")
    else:
        print("\nRun fog_cm.py first to create CM_original.csv and CM_loso.csv for the 3-panel figure.")


if __name__ == "__main__":
    main()
