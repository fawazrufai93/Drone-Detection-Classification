"""
rf_simulator.py

Synthetic RF environment generator for the AeroSense spectrum
monitoring tool.

Produces complex baseband IQ samples that emulate a slice of RF
spectrum containing:
  - a white-Gaussian noise floor
  - continuous, WiFi-like occupants (fixed center frequency, fixed
    bandwidth, near-constant power)
  - intermittent / transient occupants that switch on and off at
    random, with randomised center frequency, bandwidth and power

The generator is stateful across sweeps: continuous signals persist,
and transient signals appear/expire over a random number of sweeps.
This is what makes the waterfall history and anomaly detection in the
dashboard behave realistically over time rather than being pure
per-frame noise.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from rf_units import dbm_to_amplitude


@dataclass
class ContinuousSignal:
    """A signal present in every sweep (e.g. a fixed WiFi access point)."""
    name: str
    center_freq_hz: float
    bandwidth_hz: float
    power_dbm: float


@dataclass
class TransientSignal:
    """A signal that appears/disappears at random (e.g. a burst
    transmission, a frequency-hopping device passing through the band)."""
    name: str
    center_freq_hz: float
    bandwidth_hz: float
    power_dbm: float
    sweeps_remaining: int


class RFEnvironmentSimulator:
    """
    Generates one "sweep" of complex baseband IQ samples per call to
    `generate_sweep()`, representing a fixed span of RF spectrum
    centered on `center_freq_hz`.
    """

    def __init__(
        self,
        center_freq_hz: float = 2_441.75e6,
        span_hz: float = 83.5e6,
        sample_rate_hz: float = 84e6,
        n_samples: int = 16384,
        noise_floor_dbm: float = -95.0,
        transient_probability: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        self.center_freq_hz = center_freq_hz
        self.span_hz = span_hz
        self.sample_rate_hz = sample_rate_hz
        self.n_samples = n_samples
        self.noise_floor_dbm = noise_floor_dbm
        self.transient_probability = transient_probability

        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # Continuous occupants: real-world 2.4 GHz WiFi typically uses the
        # three non-overlapping 20 MHz channels 1, 6, 11. Two active APs
        # leaves most of the band free, which is what makes the
        # percentile-based noise floor estimate meaningful.
        self.continuous_signals: List[ContinuousSignal] = [
            ContinuousSignal("WiFi-CH1", 2_412e6, 20e6, -55.0),
            ContinuousSignal("WiFi-CH11", 2_462e6, 20e6, -62.0),
        ]
        self._active_transients: List[TransientSignal] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_sweep(self) -> np.ndarray:
        """Return one sweep of complex baseband IQ samples, shape (n_samples,)."""
        t = np.arange(self.n_samples) / self.sample_rate_hz
        iq = self._generate_noise_floor(t)

        for sig in self.continuous_signals:
            iq += self._generate_band_limited_signal(
                t, sig.center_freq_hz, sig.bandwidth_hz, sig.power_dbm
            )

        self._advance_transients()
        for sig in self._active_transients:
            iq += self._generate_band_limited_signal(
                t, sig.center_freq_hz, sig.bandwidth_hz, sig.power_dbm
            )

        return iq

    @property
    def active_transient_count(self) -> int:
        return len(self._active_transients)

    @property
    def active_transient_names(self) -> List[str]:
        return [s.name for s in self._active_transients]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _generate_noise_floor(self, t: np.ndarray) -> np.ndarray:
        noise_amp = dbm_to_amplitude(self.noise_floor_dbm)
        real = self._rng.normal(0, noise_amp / np.sqrt(2), t.size)
        imag = self._rng.normal(0, noise_amp / np.sqrt(2), t.size)
        return real + 1j * imag

    def _generate_band_limited_signal(
        self, t: np.ndarray, center_freq_hz: float, bandwidth_hz: float, power_dbm: float
    ) -> np.ndarray:
        """
        Band-limited complex noise centered at `center_freq_hz`, used as a
        stand-in for a real modulated carrier. This gives an occupant the
        right spectral footprint (a flat-ish power block over its
        bandwidth) without simulating a specific modulation scheme.
        """
        baseband_offset_hz = center_freq_hz - self.center_freq_hz
        amp = dbm_to_amplitude(power_dbm)

        n = t.size
        white = self._rng.normal(0, 1, n) + 1j * self._rng.normal(0, 1, n)
        spectrum = np.fft.fft(white)
        freq_bins = np.fft.fftfreq(n, d=1 / self.sample_rate_hz)
        band_mask = np.abs(freq_bins) <= (bandwidth_hz / 2)
        shaped = np.fft.ifft(spectrum * band_mask)

        rms = np.sqrt(np.mean(np.abs(shaped) ** 2)) + 1e-30
        shaped = shaped / rms

        carrier = np.exp(2j * np.pi * baseband_offset_hz * t)
        return amp * shaped * carrier

    def _advance_transients(self) -> None:
        # Age out / expire existing transients.
        still_active = []
        for sig in self._active_transients:
            sig.sweeps_remaining -= 1
            if sig.sweeps_remaining > 0:
                still_active.append(sig)
        self._active_transients = still_active

        # Chance of a new transient appearing this sweep.
        if self._py_rng.random() < self.transient_probability:
            margin = 2e6
            offset = self._py_rng.uniform(
                -self.span_hz / 2 + margin, self.span_hz / 2 - margin
            )
            self._active_transients.append(
                TransientSignal(
                    name=f"Transient-{self._py_rng.randint(1000, 9999)}",
                    center_freq_hz=self.center_freq_hz + offset,
                    bandwidth_hz=self._py_rng.uniform(1e6, 8e6),
                    power_dbm=self._py_rng.uniform(-70, -40),
                    sweeps_remaining=self._py_rng.randint(1, 4),
                )
            )
