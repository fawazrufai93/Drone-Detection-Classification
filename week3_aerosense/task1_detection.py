"""
TASK 1 - Detection algorithm, tested under different SNR conditions.

Run:  python task1_detection.py
Outputs (results/): task1_pd_vs_snr.png, task1_roc.png, task1_example.png,
                    task1_pd_vs_snr.csv, task1_week1_validation.csv, task1_summary.json
"""
import json, random
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aerosense import RESULTS, WEEK1_NPZ
from aerosense.detector import detect, detect_psd, psd_from_iq, segment_covers, DEFAULT_MARGIN_DB, WEEK1_MARGIN_DB
from aerosense.simulate import sample_params, synth_capture

SEED, N_TRIALS, N_NOISE = 7, 200, 2000
NOISE_FLOOR_DBM = -90.0                       # mid-range Week 2 environment (suburban)
SNR_LIST = [-6, -3, 0, 3, 6, 9, 12, 15, 20]   # dB, in-band (per-bin) SNR
CLASSES = ["UAV", "WiFi", "Generic"]
py, npr = random.Random(SEED), np.random.default_rng(SEED)
RESULTS.mkdir(exist_ok=True)

def hit(res, truth):
    return any(segment_covers(s, truth[0].center_mhz) for s in res.segments)

# ---- 1. generate trials once (store PSD so thresholds can be re-evaluated) ----------
sig_trials = {(c, s): [] for c in CLASSES for s in SNR_LIST}
for c in CLASSES:
    for s in SNR_LIST:
        for _ in range(N_TRIALS):
            p = sample_params(c, py, NOISE_FLOOR_DBM, s)
            iq, truth = synth_capture([p], NOISE_FLOOR_DBM, npr)
            f, psd = psd_from_iq(iq)
            sig_trials[(c, s)].append((f, psd, truth))
noise_trials = []
for _ in range(N_NOISE):
    iq, _ = synth_capture([], NOISE_FLOOR_DBM, npr)
    f, psd = psd_from_iq(iq); noise_trials.append((f, psd))

# ---- 1b. noise-only false alarm rate vs threshold margin (how the margin was chosen) --------
cal = pd.DataFrame([dict(margin_db=m, false_alarm_rate=np.mean([detect_psd(f, p, m).detected for f, p in noise_trials]))
                    for m in np.arange(5, 11.5, 0.5)])
cal.to_csv(RESULTS / "task1_margin_calibration.csv", index=False)
print("Noise-only false alarm rate vs margin:\n", cal.round(4).to_string(index=False))

# ---- 2. detection probability vs SNR at the default threshold -------------------------
rows = []
for (c, s), tr in sig_trials.items():
    pd_ = np.mean([hit(detect_psd(f, p, DEFAULT_MARGIN_DB), t) for f, p, t in tr])
    rows.append(dict(signal=c, snr_db=s, detection_probability=pd_))
pdf = pd.DataFrame(rows); pdf.to_csv(RESULTS / "task1_pd_vs_snr.csv", index=False)
pfa = np.mean([detect_psd(f, p, DEFAULT_MARGIN_DB).detected for f, p in noise_trials])

# confusion counts (signal-present trials = positives, noise-only trials = negatives)
TP = FN = 0
for tr in sig_trials.values():
    for f, p, t in tr:
        if hit(detect_psd(f, p, DEFAULT_MARGIN_DB), t): TP += 1
        else: FN += 1
FP = int(round(pfa * N_NOISE)); TN = N_NOISE - FP

fig, ax = plt.subplots(figsize=(7, 4.2))
for c in CLASSES:
    d = pdf[pdf.signal == c]; ax.plot(d.snr_db, d.detection_probability, "o-", label=c)
ax.axvline(DEFAULT_MARGIN_DB, color="grey", ls=":", label=f"threshold margin ({DEFAULT_MARGIN_DB:g} dB over noise floor)")
ax.set(xlabel="In-band SNR (dB)", ylabel="Detection probability Pd", ylim=(-0.02, 1.02),
       title=f"Task 1 - Pd vs SNR (noise-only false alarm rate = {pfa:.1%})")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "task1_pd_vs_snr.png", dpi=150)

# ---- 3. ROC: sweep the threshold margin (threshold detection trade-off) ---------------
margins = np.arange(3, 12, 0.5)
fig, ax = plt.subplots(figsize=(5.8, 4.5))
roc = {}
for s in (3, 6, 9):
    tr = sig_trials[("UAV", s)]
    pds = [np.mean([hit(detect_psd(f, p, m), t) for f, p, t in tr]) for m in margins]
    pfas = [np.mean([detect_psd(f, p, m).detected for f, p in noise_trials[:500]]) for m in margins]
    ax.plot(pfas, pds, "o-", ms=3, label=f"UAV, SNR {s} dB"); roc[s] = (pfas, pds)
ax.set_xscale("symlog", linthresh=2e-3)
ax.set(xlabel="False alarm rate Pfa (0 = none seen in 500 noise-only snapshots)", ylabel="Detection probability Pd",
       xlim=(0, 1), title="Task 1 - ROC (threshold margin 3-11.5 dB)")
ax.plot([], [], " ")
ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(RESULTS / "task1_roc.png", dpi=150)

# ---- 4. one worked example: PSD, noise floor, threshold, segments ---------------------
p = sample_params("UAV", random.Random(3), NOISE_FLOOR_DBM, 12)
iq, truth = synth_capture([p], NOISE_FLOOR_DBM, np.random.default_rng(3))
r = detect(iq)
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.plot(r.freqs_mhz, r.psd_db, lw=.7, label="PSD")
ax.axhline(r.noise_floor_db, color="g", ls="--", label=f"Estimated noise floor {r.noise_floor_db:.1f}")
ax.axhline(r.threshold_db, color="r", ls=":", label=f"Threshold (+{DEFAULT_MARGIN_DB:g} dB)")
for s in r.segments: ax.axvspan(s.lo_mhz, s.hi_mhz, color="orange", alpha=.3)
ax.set(xlabel="Frequency (MHz)", ylabel="PSD (dBm/Hz-equiv.)",
       title=f"Task 1 example - signal detected: {r.detected} (UAV-like, 12 dB SNR)")
ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(RESULTS / "task1_example.png", dpi=150)

# ---- 5. Week 1 recorded sweeps (ground-truth source list from rf_spectrum_monitor.py) -
d = np.load(WEEK1_NPZ)
freqs, H = d["freqs_hz"], d["psd_dbm_hz"]
SOURCES = [("WiFi ch1", 2412, "always on"), ("WiFi ch11", 2462, "always on"), ("Beacon (CW)", 2428, "always on"),
           ("Weak link", 2485, "always on"), ("Bluetooth-like", 2441, "ON 35% of dwells"),
           ("Wideband burst", 2450, "ON 20% of dwells")]
res = [detect_psd(freqs, H[i], WEEK1_MARGIN_DB) for i in range(len(H))]   # Week 1 sweeps: Week 1's own 6 dB (19 averages)
vrows = []
for name, fc, note in SOURCES:
    frac = np.mean([any(s.lo_mhz - .1 <= fc <= s.hi_mhz + .1 for s in r.segments) for r in res])
    vrows.append(dict(source=name, center_mhz=fc, expected=note, detected_in_fraction_of_sweeps=round(frac, 2)))
unexp = sum(1 for r in res for s in r.segments
            if not any(s.lo_mhz - .5 <= fc <= s.hi_mhz + .5 for _, fc, _ in SOURCES))
vdf = pd.DataFrame(vrows); vdf.to_csv(RESULTS / "task1_week1_validation.csv", index=False)

summary = dict(margin_db=DEFAULT_MARGIN_DB, week1_sweep_margin_db=WEEK1_MARGIN_DB,
               noise_only_false_alarm_rate_at_6dB=float(cal[cal.margin_db == 6.0].false_alarm_rate.iloc[0]), noise_only_false_alarm_rate=pfa, TP=TP, FN=FN, FP=FP, TN=TN,
               overall_detection_probability=TP / (TP + FN),
               snr_for_pd_ge_0p9={c: next((int(s) for s in SNR_LIST
                                           if pdf[(pdf.signal == c) & (pdf.snr_db == s)].detection_probability.iloc[0] >= .9), None)
                                  for c in CLASSES},
               week1_segments_not_matching_any_source=int(unexp), week1_sweeps=len(H))
json.dump(summary, open(RESULTS / "task1_summary.json", "w"), indent=2)
print(pdf.pivot(index="snr_db", columns="signal", values="detection_probability").round(2).to_string())
print("\nnoise-only false alarm rate:", pfa, " TP/FN/FP/TN:", TP, FN, FP, TN)
print("\nWeek 1 recorded-sweep validation:\n", vdf.to_string(index=False), "\nunmatched segments:", unexp)
print(json.dumps(summary, indent=2))
