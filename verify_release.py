import glob, os
import numpy as np
from scipy import signal
from fog_pipeline import Config, bandpass

MANIFEST = {
    "Results/results_ladder_uncertainty/LAD_T0_seed_spread.csv": "18d30f7da695888b9316f7f00c963b2a650914a5eb30a8a2a6b1e4f31155617d",
    "Results/results_ladder_uncertainty/LAD_T1_by_seed.csv": "ee93a86d3a17e28b9c7bcdd7a2d1f0101c1496881e18682f881c6d5ae7583078",
    "Results/results_ladder_uncertainty/LAD_T2_patient_bootstrap.csv": "f7e1d0c0fe40807e9b95e04d50faca63821db1eb5cdb057f9928d163bfe9bc39",
    "Results/results_ladder_uncertainty/LAD_T3_paired_differences.csv": "4cfb9afad86b8f0511c1763f9abd811204431edc7458630433b3d3ab037cf5a4",
    "Results/results_ladder_uncertainty/LAD_T4_seed42_reproduction.csv": "67308a39f299587649f6f8d81cb6dcc1816879055a39cdf1524b4fd658949ccb",
    "Results/results_mx2_external/EVS_T1_all_seeds.csv": "5987e91db2a682fa56a08ffa0d699b17ef3d238709f6430638ac15046d9f2271",
    "Results/results_mx2_external/EVS_T3_paired_differences_all_seeds.csv": "99f4d58afeffe4716e148f21c44260cc4b1b461205e616c2718c8d90f9eece54",
    "Results/results_mx2_external/EVS_T6_seed_range.csv": "5220ada9ab13038e5b61892f89ab3e9d2d50579785dffea68d92ec5df23a466f",
    "Results/results_mx2_external/EVS_choices_defog_seed1.csv": "e2c140ffbbc51bb12f22e1e5a7b87639e77dc79e03973e31c9a3573b84372eda",
    "Results/results_mx2_external/EVS_choices_defog_seed2.csv": "dcace2e6fe9df95c52d6a48f130c07123cbde996aeb3706b9ab533c2bf51f976",
    "Results/results_mx2_external/EVS_choices_defog_seed3.csv": "c29707a98193181e4d2fd36b0d412140daa178511b6ede56f8a7ba4a3dbb754a",
    "Results/results_mx2_external/EVS_choices_defog_seed4.csv": "0ecde739e828a5743bbf2e1f4754b0638b5ee9a99d2afcc9f43e5b7ad9832730",
    "Results/results_mx2_external/EVS_choices_tdcsfog_seed1.csv": "18bc6ba1d4e243df8a9ff2862074135e866def761d27c019a813cd800075a57a",
    "Results/results_mx2_external/EVS_choices_tdcsfog_seed2.csv": "342eafbdc883b1e1caf97dbce64e82e7412fd2416813811f2387b4c10120b9d7",
    "Results/results_mx2_external/EVS_choices_tdcsfog_seed3.csv": "dbae1cf4ea0245c0e8157628bea8ca7e2a5e10f4dc40a9e02070a3fa2ac7e995",
    "Results/results_mx2_external/EVS_choices_tdcsfog_seed4.csv": "695f7a3e70f04e2b932e2edfe3031b23732b18e262f57eb2bf378f680620cbc0",
    "Results/results_mx2_external/EVS_counts_defog_seed1.pkl": "05a68d2a29ca35d252577146bfd715a51db37479a8e8dda4efac7f3e2545f898",
    "Results/results_mx2_external/EVS_counts_defog_seed2.pkl": "75bca3f1ff690fd78014bda795ec237f146b4761840465d7bd0881255ef38b28",
    "Results/results_mx2_external/EVS_counts_defog_seed3.pkl": "db0dd51062fa90a0f44d97d7ee6f813cb8aef2b3645159df951cd89e057c0b8a",
    "Results/results_mx2_external/EVS_counts_defog_seed4.pkl": "cb50f0b62d458159e039e9d91309ec7595864994e05734160b4713cfdeb604a9",
    "Results/results_mx2_external/EVS_counts_tdcsfog_seed1.pkl": "23f860c6d273e529db3d007a262f8d5026d3bed4c5b72684d06881c4e4626941",
    "Results/results_mx2_external/EVS_counts_tdcsfog_seed2.pkl": "c304f4b1159f7bdb1466956b87b636243f0b2049a932562111364e2c8a29d508",
    "Results/results_mx2_external/EVS_counts_tdcsfog_seed3.pkl": "75df717fd6358a8ea37a512829dd2bf5a4a91f8428a6a63ea39ae4c23e16d1be",
    "Results/results_mx2_external/EVS_counts_tdcsfog_seed4.pkl": "fbd3e77b47efdee989840c5196db31ec00e520e8554c2c4375ac95a972ed5990",
    "Results/results_mx2_external/EV_T2_controllers_CI.csv": "5c9fa9dc361afc11e97a13661babbc9e6cd4437a01c59289392a744ab038044b",
    "Results/results_mx2_external/EV_T3_paired_differences.csv": "c47c35447c7ed71b76d0ab1896c70f6b8ce943017c22a2030f71e6eecf8597e6",
    "Results/results_mx2_external/EV_T4_hypotheses.csv": "5331248181b360778671da04ce17623229d8e3dfa3bb7023baff972277c9dae2",
    "Results/results_mx2_external/EV_T5_surrogate.csv": "4edc7fbe07b7b16b241337980c4f9c4fef4d3928af8263fd5645495c04ab85bd",
    "Results/results_mx2_external/EV_choices_defog.csv": "c3a96be7f7f12b2fa89e305e250d3e16ff8f002ae2d933562acd58459020f2e9",
    "Results/results_mx2_external/EV_choices_tdcsfog.csv": "c0d1e492d50ff91c39743428ab1581e52483cc2f9d7ba43b4756236abbb7982c",
    "Results/results_mx2_external/EV_counts_defog.pkl": "c332773cc502aeb2282f01cd5b479c1f753b672c42ef1c7e0b39d4d7dd305e6f",
    "Results/results_mx2_external/EV_counts_tdcsfog.pkl": "f41f6fc83c6124b73f67b6dd26ddc1d5868ac1e48aa0b56aa755c98abfbeb451",
    "Results/results_mx2_external/EV_per_window_defog_seed42.csv.gz": "8af2e579e093ffae588c52676d05f71d4448f6d96a3affa3b9cfcdf198477c4b",
    "Results/results_mx2_external/EV_per_window_tdcsfog_seed42.csv.gz": "bdefd53a805d0cf3baaae8ab608046733836218cb8844349ee19030f2c61b8e9",
}

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
    b1 = pd.read_csv("Results/results_constrained/B1_summary.csv", index_col=0)
    for ds, t, d in [("DAPHNET C0spec", 30.085, 89.407), ("Li2021 C0spec", 32.343, 92.079)]:
        assert abs(b1.loc[ds, "timely_pre_onset_%"] - t) < 0.01 and abs(b1.loc[ds, "detected_%"] - d) < 0.01
    print("[4] shipped results match the manuscript (timely activation 30.1% / 32.3%; detection 89.4% / 92.1%): OK")
except FileNotFoundError:
    print("[4] Results/ not found - skipped")
import hashlib
present = [f for f in MANIFEST if os.path.exists(f)]
if present:
    bad = [f for f in present if hashlib.sha256(open(f, "rb").read()).hexdigest() != MANIFEST[f]]
    assert not bad, f"result files differ from the release: {bad}"
    print(f"[5] {len(present)} of {len(MANIFEST)} external-validation and ladder result files match the release checksums: OK")
else:
    print("[5] result files not found - skipped")
print("All checks passed.")
