"""Task 1 - Detection Algorithm: RF Input -> Noise Estimation -> Detection Threshold -> Signal Detected? Yes/No.
Uses the Week 2 simulator; tested under different SNR conditions.
SNR here = total signal power / total noise power over the 84 MHz capture (power_dbm - noise_floor_dbm)."""
import random, csv
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from aerosense.sim import SIGNALS, make_capture, at_snr, noise
from aerosense.detector import Detector, psd, noise_floor

rng, py = np.random.default_rng(42), random.Random(42)
FLOOR = -92.0
det = Detector(pfa=0.01).calibrate(rng)
print(f"Thresholds (design Pfa=1%): peak={det.thr_peak:.2f}  wideband-energy={det.thr_wide:.2f}")
pfa = np.mean([det.detect(noise(FLOOR, rng))[0] for _ in range(2000)])
print(f"Measured false-alarm rate on 2000 noise-only captures: {pfa:.3%}")

snrs, TRIALS = list(range(-30, 11, 2)), 150
curves = {c: [np.mean([det.detect(make_capture([at_snr(c, s, FLOOR, py)], FLOOR, rng, py)["iq"])[0]
                       for _ in range(TRIALS)]) for s in snrs] for c in SIGNALS}
with open("results/task1_pd_vs_snr.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["snr_db"] + SIGNALS)
    w.writerows([[s] + [curves[c][i] for c in SIGNALS] for i, s in enumerate(snrs)])
print("\nSNR(dB) " + " ".join(f"{c:>9}" for c in SIGNALS))
for i, s in enumerate(snrs):
    print(f"{s:>7} " + " ".join(f"{curves[c][i]:>9.2f}" for c in SIGNALS))

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for c in SIGNALS:
    ax[0].plot(snrs, curves[c], marker="o", label=c)
ax[0].axhline(0.9, ls="--", c="gray")
ax[0].set(xlabel="Total SNR (dB)", ylabel="Detection probability Pd", title=f"Pd vs SNR (Pfa≈{pfa:.1%})")
ax[0].legend(); ax[0].grid(alpha=.3)
iq = make_capture([at_snr("UAV", -12, FLOOR, py)], FLOOR, rng, py)["iq"]
f, P = psd(iq); nf = noise_floor(P)
ax[1].plot(f / 1e6, 10 * np.log10(P)); ax[1].axhline(10 * np.log10(nf), c="g", label="noise floor estimate")
ax[1].axhline(10 * np.log10(nf * det.thr_peak), c="r", ls="--", label="detection threshold")
ax[1].set(xlabel="Baseband frequency (MHz)", ylabel="dB", title="Example: UAV-like signal at -12 dB SNR"); ax[1].legend()
plt.tight_layout(); plt.savefig("results/task1_detection.png", dpi=150)
