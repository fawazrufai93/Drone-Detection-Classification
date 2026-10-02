"""
TASK 3 - False alarm analysis: stress-test the full AeroSense engine.

Scenarios : low noise | high noise | multiple signals | overlapping signals | weak signals
Measures  : detection rate, false alarm rate, classification accuracy

Definitions (per scenario, N captures with signal(s) + N noise-only captures at the same noise floor)
  detection rate          fraction of true signals covered by a detector segment (RF activity detected)
  engine detection rate   same, but the segment must also survive classification (not "Background")
  false alarm rate        fraction of noise-only captures with RF activity detected (detector) /
                          with a reported signal (engine)
  false detections/capture  (signal captures) engine detections that explain no true signal band
  classification accuracy fraction of engine-detected true signals given the correct class label

Run:  python task3_false_alarm.py
"""
import json, random, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerosense import RESULTS
from aerosense.engine import AeroSenseEngine
from aerosense.detector import segment_covers, segment_overlaps
from aerosense.simulate import sample_params, sample_separate, sample_overlapping, synth_capture

warnings.filterwarnings("ignore")
SEED, N = 21, 300
SIGNAL_CLASSES = ["UAV", "WiFi", "Generic"]
py, npr = random.Random(SEED), np.random.default_rng(SEED)
eng = AeroSenseEngine()
RESULTS.mkdir(exist_ok=True)

class Det:  # adapter so detections work with the detector's segment helpers
    def __init__(s, d): s.lo_mhz, s.hi_mhz, s.center_mhz, s.label = d.center_mhz - d.bandwidth_mhz / 2, d.center_mhz + d.bandwidth_mhz / 2, d.center_mhz, d.label

SCENARIOS = {   # name: (noise floor dBm, how a capture is built)
    "Low noise":           (-100.0, lambda nf: [sample_params(py.choice(SIGNAL_CLASSES), py, nf)]),
    "High noise":          (-70.0,  lambda nf: [sample_params(py.choice(SIGNAL_CLASSES), py, nf)]),
    "Multiple signals":    (-90.0,  lambda nf: sample_separate([py.choice(SIGNAL_CLASSES) for _ in range(3)], py, nf)),
    "Overlapping signals": (-90.0,  lambda nf: sample_overlapping(*py.sample(SIGNAL_CLASSES, 2), py, nf)),
    "Weak signals":        (-90.0,  lambda nf: [sample_params(py.choice(SIGNAL_CLASSES), py, nf, snr_db=py.uniform(0, 10))]),
}

rows, conf_true, conf_pred = [], [], []
for name, (nf, build) in SCENARIOS.items():
    n_sig = n_det = n_edet = n_cls_ok = n_false = 0
    for _ in range(N):
        plist = build(nf)
        iq, truth = synth_capture(plist, nf, npr)
        res = eng.process(iq); dets = [Det(d) for d in res.detections]
        for t in truth:
            n_sig += 1
            n_det += any(segment_covers(sg, t.center_mhz) for sg in res.segments)
            hit = [d for d in dets if segment_covers(d, t.center_mhz)]
            if hit:
                n_edet += 1; best = min(hit, key=lambda d: abs(d.center_mhz - t.center_mhz))
                n_cls_ok += best.label == t.cls; conf_true.append(t.cls); conf_pred.append(best.label)
            else:
                conf_true.append(t.cls); conf_pred.append("Missed")
        bands = [b for t in truth for b in t.bands()]
        n_false += sum(not any(segment_overlaps(d, lo, hi) for lo, hi in bands) for d in dets)
    fa_det = fa = 0
    for _ in range(N):  # noise-only captures at the same noise floor
        iq, _t = synth_capture([], nf, npr)
        r = eng.process(iq); fa_det += r.rf_activity; fa += len(r.detections) > 0
    rows.append(dict(scenario=name, noise_floor_dbm=nf, true_signals=n_sig,
                     detection_rate=n_det / n_sig, engine_detection_rate=n_edet / n_sig,
                     false_alarm_rate=fa_det / N, engine_false_alarm_rate=fa / N,
                     false_detections_per_capture=n_false / N,
                     classification_accuracy=n_cls_ok / max(n_edet, 1)))
    print(rows[-1])

df = pd.DataFrame(rows).round(3); df.to_csv(RESULTS / "task3_results.csv", index=False)
print("\n", df.to_string(index=False))

# ---- figures -------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.2)); w = .25; x = np.arange(len(df))
for i, (col, lab) in enumerate([("detection_rate", "Detection rate (detector)"), ("engine_detection_rate", "Detection rate (engine)"),
                                ("false_alarm_rate", "False alarm rate (noise only)"), ("classification_accuracy", "Classification accuracy")]):
    bars = ax.bar(x + (i - 1.5) * 0.2, df[col], 0.2, label=lab)
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in df[col]], fontsize=6.5, padding=1)
ax.set_xticks(x); ax.set_xticklabels(df.scenario, rotation=12); ax.set(ylim=(0, 1.3), title="Task 3 - AeroSense stress test")
ax.legend(loc="upper right", fontsize=8, ncol=2); ax.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(RESULTS / "task3_scenarios.png", dpi=150)

labels_t, labels_p = ["UAV", "WiFi", "Generic"], ["UAV", "WiFi", "Generic", "Background", "Missed"]
cm = np.array([[sum(1 for a, b in zip(conf_true, conf_pred) if a == t and b == p) for p in labels_p] for t in labels_t])
fig, ax = plt.subplots(figsize=(6.4, 4)); ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels_p))); ax.set_xticklabels(labels_p); ax.set_yticks(range(len(labels_t))); ax.set_yticklabels(labels_t)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set(title="Task 3 - true vs engine output, all scenarios", ylabel="True class", xlabel="Engine output")
fig.tight_layout(); fig.savefig(RESULTS / "task3_confusion_matrix.png", dpi=150)
json.dump(df.to_dict("records"), open(RESULTS / "task3_summary.json", "w"), indent=2)
