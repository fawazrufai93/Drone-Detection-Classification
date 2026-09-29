"""
spectrum_analyzer.py

DSP pipeline for the AeroSense spectrum monitoring tool:

    IQ samples -> Welch PSD -> noise-floor estimate -> occupancy map

Noise floor estimation uses a percentile method: because occupied
spectrum is (by definition) a minority of most real bands, a low
percentile (default 20th) of the power distribution across bins is a
robust estimate of the noise floor even without knowing in advance
which bins are occupied. Bins whose power exceeds the noise floor by
more than `occupancy_threshold_db` are classified Occupied.

Welch's method (scipy.signal.welch) reports power *spectral density*
(power per Hz), whose absolute scale depends on the window function,
segment length and sample rate. Rather than hand-deriving that scale
factor, the analyzer self-calibrates once at construction time: it
runs a synthetic noise reference of known dBm through its own
pipeline and stores the offset needed to make Welch's output read
that same dBm back out. This is the same idea a real spectrum
analyzer uses when it calibrates against a reference source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import signal

from rf_units import dbm_to_amplitude


@dataclass
class OccupancyBin:
    frequency_mhz: float
    power_dbm: float
    occupied: bool


@dataclass
class SweepResult:
    frequencies_mhz: np.ndarray
    power_dbm: np.ndarray
    noise_floor_dbm: float
    occupancy: List[OccupancyBin]
    occupancy_percentage: float


class SpectrumAnalyzer:
    def __init__(
        self,
        sample_rate_hz: float,
        center_freq_hz: float,
        nperseg: int = 512,
        occupancy_threshold_db: float = 6.0,
        noise_floor_percentile: float = 20.0,
        calibration_n_samples: int = 8192,
        calibration_reference_dbm: float = -80.0,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.center_freq_hz = center_freq_hz
        self.nperseg = nperseg
        self.occupancy_threshold_db = occupancy_threshold_db
        self.noise_floor_percentile = noise_floor_percentile
        self._calibration_offset_db = self._calibrate(
            calibration_n_samples, calibration_reference_dbm
        )

    def _calibrate(self, n_samples: int, reference_dbm: float) -> float:
        """Measure this analyzer's raw Welch-PSD offset against a
        synthetic noise reference of known power, so `analyze()` can
        report physically meaningful dBm regardless of window/nperseg."""
        rng = np.random.default_rng(0)
        amp = dbm_to_amplitude(reference_dbm)
        reference_iq = amp * (
            rng.normal(0, 1 / np.sqrt(2), n_samples)
            + 1j * rng.normal(0, 1 / np.sqrt(2), n_samples)
        )
        _, psd = signal.welch(
            reference_iq,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=min(self.nperseg, n_samples),
            return_onesided=False,
            scaling="density",
        )
        measured_dbm = 10 * np.log10(np.median(psd) + 1e-20)
        return reference_dbm - measured_dbm

    def analyze(self, iq_samples: np.ndarray) -> SweepResult:
        """Run one sweep of IQ samples through the full PSD -> noise
        floor -> occupancy pipeline."""
        freq_bins_hz, psd = signal.welch(
            iq_samples,
            fs=self.sample_rate_hz,
            window="hann",
            nperseg=min(self.nperseg, len(iq_samples)),
            return_onesided=False,
            scaling="density",
        )

        order = np.argsort(freq_bins_hz)
        freq_bins_hz = freq_bins_hz[order]
        psd = psd[order]

        power_dbm = self._psd_to_dbm(psd)
        frequencies_mhz = (self.center_freq_hz + freq_bins_hz) / 1e6

        noise_floor_dbm = float(np.percentile(power_dbm, self.noise_floor_percentile))
        occupied_mask = power_dbm > (noise_floor_dbm + self.occupancy_threshold_db)

        occupancy = [
            OccupancyBin(frequency_mhz=float(f), power_dbm=float(p), occupied=bool(o))
            for f, p, o in zip(frequencies_mhz, power_dbm, occupied_mask)
        ]
        occupancy_percentage = float(100.0 * occupied_mask.mean())

        return SweepResult(
            frequencies_mhz=frequencies_mhz,
            power_dbm=power_dbm,
            noise_floor_dbm=noise_floor_dbm,
            occupancy=occupancy,
            occupancy_percentage=occupancy_percentage,
        )

    def _psd_to_dbm(self, psd: np.ndarray) -> np.ndarray:
        """Convert a linear power-spectral-density array to calibrated dBm."""
        return 10 * np.log10(psd + 1e-20) + self._calibration_offset_db
