"""Run AeroSense RF Detection Engine v1 on a few example captures.   Run: python aerosense_demo.py"""
import random, numpy as np
from aerosense import RESULTS
from aerosense.engine import AeroSenseEngine
from aerosense.simulate import sample_params, sample_separate, synth_capture

eng = AeroSenseEngine()
py, npr = random.Random(11), np.random.default_rng(11)
cases = [
    ("UAV-like signal",                        [sample_params("UAV", py)]),
    ("Wi-Fi-like signal",                      [sample_params("WiFi", py)]),
    ("Generic narrowband signal",              [sample_params("Generic", py)]),
    ("Background noise only",                  [sample_params("Background", py)]),
    ("UAV-like + Wi-Fi-like (separate bands)", sample_separate(["UAV", "WiFi"], py, -90.0, min_gap_mhz=8.0)),
]
lines = [f"AeroSense RF Detection Engine v1  (classifier: {eng.model_name}, alert at UAV confidence >= {eng.alert_confidence})\n"]
for title, plist in cases:
    nf = plist[0].noise_floor_dbm               # Week 2 environment noise floor for this capture
    iq, truth = synth_capture(plist, nf, npr)
    lines += [f"=== {title}  | truth: {[(t.cls, round(t.center_mhz,1)) for t in truth] or 'noise only'}", eng.process(iq).report(), ""]
txt = "\n".join(lines); print(txt); (RESULTS / "engine_demo.txt").write_text(txt)
