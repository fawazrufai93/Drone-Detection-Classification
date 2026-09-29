"""Task 3 - False Alarm Analysis: low noise, high noise, multiple signals, overlapping signals, weak signals.
Measures: detection rate, false alarm rate, classification accuracy (end-to-end through the engine)."""
import random, csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace
from aerosense.engine import AeroSenseEngine
from aerosense.sim import SIGNALS, make_capture, profile_signal, at_snr

eng = AeroSenseEngine(seed=3).fit()
rng, py = np.random.default_rng(99), random.Random(99)
T = 300

def one(floor, kind="awgn"):
    c = py.choice(SIGNALS); return [profile_signal(c, floor, py)], {c}, floor, kind

def multi(floor):
    cs = py.sample(SIGNALS, py.choice([2, 3])); return [profile_signal(c, floor, py) for c in cs], set(cs), floor, "awgn"

def overlap(floor):
    w = profile_signal("WiFi", floor, py); n = py.choice(["UAV", "Generic"])
    p = replace(profile_signal(n, floor, py), center_freq_hz=w.center_freq_hz + py.uniform(-5e6, 5e6))
    return [w, p], {"WiFi", n}, floor, "awgn"

def weak(floor):
    c = py.choice(SIGNALS); return [at_snr(c, -8, floor, py)], {c}, floor, "awgn"

scen = {
    "Low noise":               lambda: one(-100.0),
    "High noise":              lambda: one(-78.0),
    "High noise + interf.":    lambda: one(-78.0, "congested"),
    "Multiple signals":        lambda: multi(-92.0),
    "Overlapping signals":     lambda: overlap(-92.0),
    "Weak signals (-8 dB)":    lambda: weak(-92.0),
}
res = []
print(f"{'Scenario':<24}{'Detection':>10}{'FalseAlarm':>12}{'ClassAcc':>10}")
for name, gen in scen.items():
    hit = ok = fa = 0
    for _ in range(T):
        pl, truth, floor, kind = gen()
        o = eng.process(make_capture(pl, floor, rng, py, kind))
        hit += o["rf_activity_detected"]; ok += o["classification"] in truth
        fa += eng.process(make_capture([], floor, rng, py, kind))["rf_activity_detected"]   # noise-only, same conditions
    res.append((name, hit / T, fa / T, ok / T))
    print(f"{name:<24}{hit/T:>10.1%}{fa/T:>12.1%}{ok/T:>10.1%}")
with open("results/task3_false_alarm.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["scenario", "detection_rate", "false_alarm_rate", "classification_accuracy"]); w.writerows(res)

x = np.arange(len(res)); fig, ax = plt.subplots(figsize=(12, 4.5))
for i, (lab, k) in enumerate([("Detection rate", 1), ("False alarm rate", 2), ("Classification accuracy", 3)]):
    ax.bar(x + (i - 1) * .27, [r[k] for r in res], .27, label=lab)
ax.set_xticks(x); ax.set_xticklabels([r[0] for r in res], rotation=12); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=.3)
ax.set_title("False alarm analysis"); plt.tight_layout(); plt.savefig("results/task3_false_alarm.png", dpi=150)
