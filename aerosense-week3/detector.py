"""
detector.py  --  Week 3, Task 1: Detection Algorithm

    RF Input -> Noise Estimation -> Detection Threshold -> Signal Detected? (Yes/No)

Re-uses the Week 1 tools:
  * RFEnvironmentSimulator  (rf_simulator.py)  -> RF input (IQ samples)
  * SpectrumAnalyzer        (spectrum_analyzer.py) -> Welch PSD + noise-floor estimate

Detection rule (energy / peak detection on the spectrum):
    threshold_dbm = estimated_noise_floor_dbm + margin_db
    detected      = at least `min_bins` PSD bins are above threshold_dbm

SNR convention used for the tests:
    SNR = in-band signal PSD - noise PSD  (dB, per frequency bin)
i.e. how far the signal stands above the noise floor on the spectrum plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rf_simulator import RFEnvironmentSimulator
from spectrum_analyzer import SpectrumAnalyzer

CENTER_FREQ_HZ = 2_441.75e6
SAMPLE_RATE_HZ = 84e6
N_SAMPLES = 16384


@dataclass
class DetectionResult:
    detected: bool              # Signal Detected?  Yes / No
    noise_floor_dbm: float      # Noise Estimation
    threshold_dbm: float        # Detection Threshold
    peak_power_dbm: float       # strongest bin
    bins_above: int             # bins over threshold

    @property
    def label(self) -> str:
        return "Yes" if self.detected else "No"


class SignalDetector:
    def __init__(self, margin_db: float = 3.0, min_bins: int = 3) -> None:
        self.margin_db = margin_db
        self.min_bins = min_bins
        self.analyzer = SpectrumAnalyzer(SAMPLE_RATE_HZ, CENTER_FREQ_HZ)

    def detect(self, iq: np.ndarray) -> DetectionResult:
        sweep = self.analyzer.analyze(iq)                      # RF input -> PSD
        noise = sweep.noise_floor_dbm                          # noise estimation
        threshold = noise + self.margin_db                     # detection threshold
        above = int(np.sum(sweep.power_dbm > threshold))
        return DetectionResult(
            detected=above >= self.min_bins,                   # Yes / No
            noise_floor_dbm=noise,
            threshold_dbm=threshold,
            peak_power_dbm=float(sweep.power_dbm.max()),
            bins_above=above,
        )


def make_iq(noise_floor_dbm: float, snr_db: Optional[float], bandwidth_hz: float = 2e6,
            center_freq_hz: float = 2_445e6, seed: Optional[int] = None) -> np.ndarray:
    """One sweep of IQ. snr_db=None -> noise only. Uses the Week 1 simulator
    with its default WiFi/transient occupants switched off so SNR is controlled."""
    sim = RFEnvironmentSimulator(
        center_freq_hz=CENTER_FREQ_HZ, sample_rate_hz=SAMPLE_RATE_HZ, n_samples=N_SAMPLES,
        noise_floor_dbm=noise_floor_dbm, transient_probability=0.0, seed=seed,
    )
    sim.continuous_signals = []
    iq = sim.generate_sweep()
    if snr_db is not None:
        # convert per-bin SNR to total signal power over its bandwidth
        power_dbm = noise_floor_dbm + snr_db - 10 * np.log10(SAMPLE_RATE_HZ / bandwidth_hz)
        t = np.arange(N_SAMPLES) / SAMPLE_RATE_HZ
        iq = iq + sim._generate_band_limited_signal(t, center_freq_hz, bandwidth_hz, power_dbm)
    return iq


def run_snr_test(snr_levels=range(-6, 13, 2), trials: int = 200,
                 noise_floor_dbm: float = -92.0, margin_db: float = 3.0, min_bins: int = 3):
    det = SignalDetector(margin_db, min_bins)
    # false alarm rate: noise-only sweeps
    fa = sum(det.detect(make_iq(noise_floor_dbm, None, seed=10_000 + i)).detected
             for i in range(trials)) / trials
    rows = []
    for snr in snr_levels:
        hits = sum(det.detect(make_iq(noise_floor_dbm, snr, seed=i)).detected
                   for i in range(trials))
        rows.append((snr, hits / trials))
    return fa, rows


if __name__ == "__main__":
    import sys
    margin = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    min_bins = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    fa, rows = run_snr_test(margin_db=margin, min_bins=min_bins)
    print(f"Detector: threshold = noise floor + {margin} dB, min_bins = {min_bins}")
    print(f"False alarm rate (noise only, 200 trials): {fa:.3f}\n")
    print(" SNR (dB) | Detection probability (Pd)")
    print("----------+---------------------------")
    for snr, pd in rows:
        print(f"  {snr:>6}  |  {pd:.3f}")
