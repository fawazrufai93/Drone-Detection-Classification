"""
pipeline.py

Implements the Week 2 signal processing pipeline:

    Synthetic Signal -> Noise -> FFT -> Filtering -> Feature Extraction
        -> Feature Vector -> Signal Class

Two parallel branches feed the final feature vector:

1. Spectral branch: synthesize a short complex-IQ snapshot of the
   signal (or pure noise, for the Background class), run it through
   Welch's method to get a calibrated power spectral density, apply a
   bandpass filter around the detected occupied region to refine the
   estimate, then extract center frequency, bandwidth, power, SNR,
   spectral flatness and PAPR from it.

2. Temporal branch: simulate a short on/off amplitude envelope over a
   longer time window using the signal's burst timing parameters,
   then measure duration and burst interval back out of that envelope
   via run-length thresholding - i.e. these are *measured* features,
   not just the input parameters copied through.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal

from signal_profiles import SignalParams

# --- Spectral snapshot configuration ----------------------------------------
ANALYSIS_CENTER_FREQ_HZ = 2_441.75e6
ANALYSIS_SAMPLE_RATE_HZ = 84e6
ANALYSIS_N_SAMPLES = 8192
ANALYSIS_NPERSEG = 2048

# --- Temporal envelope configuration ----------------------------------------
ENVELOPE_WINDOW_S = 2.0
ENVELOPE_SAMPLE_RATE_HZ = 1000.0  # 1 ms resolution


def dbm_to_amplitude(power_dbm: float) -> float:
    power_mw = 10 ** (power_dbm / 10)
    return float(np.sqrt(power_mw / 1000))


_CALIBRATION_OFFSET_DB: Optional[float] = None


def _get_calibration_offset() -> float:
    """Self-calibrate the Welch-PSD -> dBm conversion once, against a
    synthetic noise reference of known power (same approach as the
    Week 1 spectrum analyzer)."""
    global _CALIBRATION_OFFSET_DB
    if _CALIBRATION_OFFSET_DB is not None:
        return _CALIBRATION_OFFSET_DB

    reference_dbm = -80.0
    rng = np.random.default_rng(0)
    amp = dbm_to_amplitude(reference_dbm)
    ref_iq = amp * (
        rng.normal(0, 1 / np.sqrt(2), ANALYSIS_N_SAMPLES)
        + 1j * rng.normal(0, 1 / np.sqrt(2), ANALYSIS_N_SAMPLES)
    )
    _, psd = signal.welch(
        ref_iq,
        fs=ANALYSIS_SAMPLE_RATE_HZ,
        window="hann",
        nperseg=min(ANALYSIS_NPERSEG, ANALYSIS_N_SAMPLES),
        return_onesided=False,
        scaling="density",
    )
    measured_dbm = 10 * np.log10(np.median(psd) + 1e-20)
    _CALIBRATION_OFFSET_DB = reference_dbm - measured_dbm
    return _CALIBRATION_OFFSET_DB


def _generate_bandlimited_noise(n_samples: int, bandwidth_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Generate complex baseband noise low-pass filtered to
    bandwidth_hz/2, using an actual IIR filter (smooth roll-off)
    rather than an ideal brick-wall FFT mask. An ideal brick-wall mask
    applied over the whole block has a sinc-shaped time-domain kernel;
    once the result is re-analyzed through Welch's much shorter
    per-segment FFTs (a different, coarser frequency grid), that sinc
    ringing shows up as leakage smeared across the spectrum.

    The IIR filter itself starts from a zero initial state and takes
    some samples to settle - that startup transient is broadband and,
    if left in, shows up the same way once Welch segments the signal.
    So we generate extra warm-up samples and discard them, keeping
    only the steady-state portion."""
    warmup = 512
    nyquist_hz = ANALYSIS_SAMPLE_RATE_HZ / 2
    cutoff = min(max(bandwidth_hz / 2 / nyquist_hz, 1e-4), 0.99)
    b, a = signal.butter(4, cutoff, btype="low")
    white = rng.normal(0, 1, n_samples + warmup) + 1j * rng.normal(0, 1, n_samples + warmup)
    filtered = signal.lfilter(b, a, white)[warmup:]
    rms = np.sqrt(np.mean(np.abs(filtered) ** 2)) + 1e-30
    return filtered / rms


def synthesize_spectral_snapshot(params: SignalParams, rng: np.random.Generator) -> np.ndarray:
    """Generate one complex-IQ snapshot: noise floor, plus a
    band-limited signal component if this isn't the Background class."""
    n = ANALYSIS_N_SAMPLES
    t = np.arange(n) / ANALYSIS_SAMPLE_RATE_HZ

    noise_amp = dbm_to_amplitude(params.noise_floor_dbm)
    iq = noise_amp / np.sqrt(2) * (
        rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n)
    )

    if params.center_freq_hz is None:
        return iq  # Background: noise only

    baseband_offset_hz = params.center_freq_hz - ANALYSIS_CENTER_FREQ_HZ
    amp = dbm_to_amplitude(params.power_dbm)

    shaped = _generate_bandlimited_noise(n, params.bandwidth_hz, rng)

    carrier = np.exp(2j * np.pi * baseband_offset_hz * t)
    iq = iq + amp * shaped * carrier
    return iq


def compute_psd(iq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies_mhz, power_dbm) via calibrated Welch PSD."""
    offset_db = _get_calibration_offset()
    freq_bins_hz, psd = signal.welch(
        iq,
        fs=ANALYSIS_SAMPLE_RATE_HZ,
        window="hann",
        nperseg=min(ANALYSIS_NPERSEG, len(iq)),
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freq_bins_hz)
    freq_bins_hz = freq_bins_hz[order]
    psd = psd[order]
    power_dbm = 10 * np.log10(psd + 1e-20) + offset_db
    frequencies_mhz = (ANALYSIS_CENTER_FREQ_HZ + freq_bins_hz) / 1e6
    return frequencies_mhz, power_dbm


def bandpass_filter(iq: np.ndarray, center_freq_hz: float, bandwidth_hz: float) -> np.ndarray:
    """Isolate the detected occupied region before refining the
    measurement - the 'Filtering' stage of the pipeline. Implemented
    as a frequency-domain mask (equivalent to an ideal bandpass) since
    the signal is represented as baseband-relative complex samples."""
    n = len(iq)
    freq_bins = np.fft.fftfreq(n, d=1 / ANALYSIS_SAMPLE_RATE_HZ)
    baseband_offset_hz = center_freq_hz - ANALYSIS_CENTER_FREQ_HZ
    mask = np.abs(freq_bins - baseband_offset_hz) <= (bandwidth_hz / 2 + 0.5e6)
    spectrum = np.fft.fft(iq)
    return np.fft.ifft(spectrum * mask)


def _largest_contiguous_run(mask: np.ndarray, power_linear: np.ndarray) -> np.ndarray:
    """Given a boolean occupancy mask over frequency-sorted bins,
    return a mask keeping only the highest-total-power contiguous run
    of True values. Using the full min/max span of every bin that
    crosses the threshold is not robust: a single stray noise spike
    far from the real signal (inevitable with a finite number of
    Welch segments) would blow up the bandwidth measurement. Real
    spectrum analyzers report occupied bandwidth from the main lobe,
    not scattered isolated detections."""
    n = len(mask)
    best_start, best_end, best_power = 0, 0, -1.0
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        run_power = float(np.sum(power_linear[i:j]))
        if run_power > best_power:
            best_power = run_power
            best_start, best_end = i, j
        i = j
    result = np.zeros(n, dtype=bool)
    result[best_start:best_end] = True
    return result


def extract_spectral_features(
    frequencies_mhz: np.ndarray, power_dbm: np.ndarray, iq_snapshot: np.ndarray
) -> dict:
    """Center frequency, occupied bandwidth, power, SNR, spectral
    flatness and PAPR - all measured from the PSD / IQ snapshot."""
    noise_floor_dbm = float(np.percentile(power_dbm, 15))
    raw_occupied_mask = power_dbm > (noise_floor_dbm + 6.0)

    if not raw_occupied_mask.any():
        # Nothing rose above the noise floor threshold - background-like.
        power_linear = 10 ** (power_dbm / 10)
        spectral_flatness = float(
            np.exp(np.mean(np.log(power_linear + 1e-30))) / (np.mean(power_linear) + 1e-30)
        )
        peak = np.max(np.abs(iq_snapshot))
        avg = np.sqrt(np.mean(np.abs(iq_snapshot) ** 2)) + 1e-30
        papr_db = float(20 * np.log10(peak / avg + 1e-30))
        return {
            "center_freq_mhz": None,
            "bandwidth_mhz": None,
            "power_dbm": round(noise_floor_dbm, 2),
            "snr_db": 0.0,
            "spectral_flatness": round(spectral_flatness, 4),
            "papr_db": round(papr_db, 2),
        }

    power_linear_all_bins = 10 ** (power_dbm / 10)
    occupied_mask = _largest_contiguous_run(raw_occupied_mask, power_linear_all_bins)

    occ_freqs = frequencies_mhz[occupied_mask]
    occ_power_dbm = power_dbm[occupied_mask]
    occ_power_linear = 10 ** (occ_power_dbm / 10)

    center_freq_mhz = float(np.sum(occ_freqs * occ_power_linear) / np.sum(occ_power_linear))
    bandwidth_mhz = float(occ_freqs.max() - occ_freqs.min())
    total_power_dbm = float(10 * np.log10(np.sum(occ_power_linear) + 1e-20))
    snr_db = float(total_power_dbm - noise_floor_dbm)

    power_linear_all = 10 ** (power_dbm / 10)
    spectral_flatness = float(
        np.exp(np.mean(np.log(power_linear_all + 1e-30))) / (np.mean(power_linear_all) + 1e-30)
    )

    peak = np.max(np.abs(iq_snapshot))
    avg = np.sqrt(np.mean(np.abs(iq_snapshot) ** 2)) + 1e-30
    papr_db = float(20 * np.log10(peak / avg + 1e-30))

    return {
        "center_freq_mhz": round(center_freq_mhz, 3),
        "bandwidth_mhz": round(bandwidth_mhz, 3),
        "power_dbm": round(total_power_dbm, 2),
        "snr_db": round(snr_db, 2),
        "spectral_flatness": round(spectral_flatness, 4),
        "papr_db": round(papr_db, 2),
    }


def extract_burst_features(params: SignalParams, rng: random.Random) -> dict:
    """Simulate a short on/off amplitude envelope from the signal's
    burst timing, then measure duration and burst interval back out
    of it via run-length thresholding, rather than just echoing the
    input parameters."""
    n_env = int(ENVELOPE_WINDOW_S * ENVELOPE_SAMPLE_RATE_HZ)
    t = np.arange(n_env) / ENVELOPE_SAMPLE_RATE_HZ

    if params.burst_duration_s is None:
        return {"duration_s": 0.0, "burst_interval_s": None}

    period = max(params.burst_period_s, params.burst_duration_s)
    phase = rng.uniform(0, period)
    envelope = ((t + phase) % period) < params.burst_duration_s

    # Run-length encode the envelope to measure on-durations and gaps
    # between run starts.
    changes = np.flatnonzero(np.diff(envelope.astype(int)) != 0)
    edges = np.concatenate(([0], changes + 1, [n_env]))

    on_durations = []
    run_starts = []
    for i in range(len(edges) - 1):
        start, end = edges[i], edges[i + 1]
        if envelope[start]:
            on_durations.append((end - start) / ENVELOPE_SAMPLE_RATE_HZ)
            run_starts.append(start / ENVELOPE_SAMPLE_RATE_HZ)

    if not on_durations:
        # The window was shorter than one period - treat as continuous.
        return {
            "duration_s": round(float(ENVELOPE_WINDOW_S), 4),
            "burst_interval_s": None,
        }

    duration_measured = float(np.mean(on_durations))
    if len(run_starts) > 1:
        gaps = np.diff(run_starts)
        interval_measured = float(np.mean(gaps))
    else:
        interval_measured = None

    return {
        "duration_s": round(duration_measured, 4),
        "burst_interval_s": round(interval_measured, 4) if interval_measured is not None else None,
    }


def build_feature_row(params: SignalParams, np_rng: np.random.Generator, py_rng: random.Random) -> dict:
    """Run one signal instance through the full pipeline and return
    its feature vector as a flat dict (one dataset row)."""
    iq = synthesize_spectral_snapshot(params, np_rng)
    frequencies_mhz, power_dbm = compute_psd(iq)

    if params.center_freq_hz is not None:
        filtered_iq = bandpass_filter(iq, params.center_freq_hz, params.bandwidth_hz)
        freq_mhz_filtered, power_dbm_filtered = compute_psd(filtered_iq)
        spectral = extract_spectral_features(freq_mhz_filtered, power_dbm_filtered, filtered_iq)
    else:
        spectral = extract_spectral_features(frequencies_mhz, power_dbm, iq)

    burst = extract_burst_features(params, py_rng)

    row = {
        "signal_type": params.signal_class,
        "environment": params.environment,
        "frequency": spectral["center_freq_mhz"],
        "bandwidth": spectral["bandwidth_mhz"],
        "power": spectral["power_dbm"],
        "snr": spectral["snr_db"],
        "duration": burst["duration_s"],
        "burst_interval": burst["burst_interval_s"],
        "spectral_flatness": spectral["spectral_flatness"],
        "papr_db": spectral["papr_db"],
        "noise_floor_dbm": round(params.noise_floor_dbm, 2),
    }
    return row
