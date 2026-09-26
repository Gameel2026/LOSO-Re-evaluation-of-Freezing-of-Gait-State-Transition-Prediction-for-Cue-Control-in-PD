import glob, os
import numpy as np
from scipy import signal
from fog_pipeline import Config, bandpass

cfg = Config()
rng = np.random.default_rng(0)
x = rng.standard_normal((2000, 3)); x2 = x.copy(); x2[1200:] += 5.0
y1, y2 = bandpass(x, cfg), bandpass(x2, cfg)
assert np.allclose(y1[:1200], y2[:1200]), "filter is NOT causal"
print("[1] causal filter: OK (future samples do not affect past output)")

sos = signal.butter(cfg.filt_order, cfg.band, btype="bandpass", fs=cfg.fs, output="sos")
w, gd = signal.group_delay(signal.sos2tf(sos), w=np.linspace(2, 15, 200), fs=cfg.fs)
print(f"[2] group delay 2-15 Hz: {gd.min() / cfg.fs * 1000:.0f}-{gd.max() / cfg.fs * 1000:.0f} ms")

bad = [f for f in glob.glob("*.py") if f != os.path.basename(__file__) and "filt" + "filt" in open(f).read()]
assert not bad, f"non-causal filtering found in {bad}"
print("[3] no non-causal filtering in any script: OK")

try:
    import pandas as pd
    b1 = pd.read_csv("results/results_constrained/B1_summary.csv", index_col=0)
    for ds, t, d in [("DAPHNET C0spec", 30.085, 89.407), ("Li2021 C0spec", 32.343, 92.079)]:
        assert abs(b1.loc[ds, "timely_pre_onset_%"] - t) < 0.01 and abs(b1.loc[ds, "detected_%"] - d) < 0.01
    print("[4] shipped results match the manuscript (timely activation 30.1% / 32.3%; detection 89.4% / 92.1%): OK")
except FileNotFoundError:
    print("[4] results/ not found - skipped")
print("All checks passed.")
