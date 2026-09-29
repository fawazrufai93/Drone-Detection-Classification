"""Deliverable: AeroSense RF Detection Engine v1
RF Activity Detected -> Signal Features -> Classification -> Confidence -> Alert"""
import random
import numpy as np
from aerosense import AeroSenseEngine
from aerosense.sim import make_capture, profile_signal, at_snr
from aerosense.features import FEATURES

eng = AeroSenseEngine(seed=1).fit()
eng.save("results/aerosense_engine_v1.joblib")
rng, py = np.random.default_rng(2024), random.Random(2024)
F = -92.0
cases = [("background noise only", [], F), ("WiFi (profile power)", [profile_signal("WiFi", F, py)], F),
         ("Generic comms (profile power)", [profile_signal("Generic", F, py)], F),
         ("UAV-like (profile power)", [profile_signal("UAV", F, py)], F),
         ("UAV-like in high noise (-78 dBm floor)", [profile_signal("UAV", -78.0, py)], -78.0),
         ("UAV-like, weak (-12 dB SNR)", [at_snr("UAV", -12, F, py)], F)]
print("AeroSense RF Detection Engine v1\n" + "=" * 60)
for name, pl, floor in cases:
    o = eng.process(make_capture(pl, floor, rng, py))
    print(f"\n[input: {name}]\n  RF Activity Detected : {o['rf_activity_detected']}")
    if o["features"] is not None:
        print("  Signal Features      : " + ", ".join(f"{k}={v:.3g}" for k, v in zip(FEATURES, o["features"])))
        print(f"  Classification       : {o['classification']}\n  Confidence           : {o['confidence']:.2f}")
    print(f"  Alert                : {o['alert']}")
