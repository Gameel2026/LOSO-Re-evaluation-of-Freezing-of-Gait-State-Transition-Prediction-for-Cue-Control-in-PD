import fog_rerun_causal as rc
import fog_fix2 as f2
import fog_improve as fi
import fog_final_c1 as fc

from paths import LI_CONVERTED as LI
STAGES = {"CAUSAL": True, "FIX2": True, "IMPROVE": True, "C1_FINAL": True}

rc.ROOT, rc.OUT = LI + "_shank", "results_li_causal"
rc.SENSOR_ROOTS = {"Shank": LI + "_shank", "Trunk": LI + "_trunk", "Shank+waist": LI + "_shankwaist"}
rc.SENSORS = {"Shank": ("ank",), "Trunk": ("tr",), "Shank+waist": ("ank", "tr")}
f2.OUT = "results_li_causal"
fi.SRC, fi.OUT = "results_li_causal", "results_li_improve"
fi.SENSORS = {"Shank": "Shank", "Trunk": "Trunk", "Shank+waist": "Shank+waist"}

if __name__ == "__main__":
    if STAGES["CAUSAL"]:
        print("\n=== E1-E5 (causal) on Li 2021 ==="); rc.main()
    if STAGES["FIX2"]:
        print("\n=== P0 real-only + surrogate on Li 2021 ==="); f2.main()
    if STAGES["IMPROVE"]:
        print("\n=== Ablation C0-C3 on Li 2021 ==="); fi.RUN = {"ABLATION": True, "SENSORS": False, "STATS": True}; fi.main()
    if STAGES["C1_FINAL"]:
        print("\n=== C1 sensors, surrogate, statistics on Li 2021 ===")
        fc.SENSORS_C1 = [("Trunk", "Trunk"), ("Shank+waist", "Shank+waist")]
        fc.main()
    print("\nAll Li 2021 analyses done.")
