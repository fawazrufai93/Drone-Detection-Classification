"""
signal_profiles.py

Defines the four signal classes from the Week 2 brief and the
parameter distributions each is drawn from. These are not arbitrary:
they're chosen to reflect how each class actually behaves in the RF
domain, which is what makes the classification problem meaningful
(the classes should be separable by their RF characteristics, not
just by a fixed label).

Class A - UAV-like:      narrow instantaneous bandwidth, frequency-
                          hopping style bursts, short duration, fast
                          repetition - similar to FHSS drone control
                          links.
Class B - WiFi-like:     wide (20/40 MHz) channels centered on the
                          standard non-overlapping WiFi channels,
                          longer packet-style bursts.
Class C - Generic comms: narrowband, near-continuous carrier (e.g.
                          telemetry / LMR-style), long duration, low
                          burst repetition.
Class D - Background:    no synthesized signal at all - pure noise.

All four classes share the same swept frequency span used in the
Week 1 tool (2400-2483.5 MHz) so that class separation in the dataset
comes from bandwidth/power/timing characteristics rather than simply
from frequency location, matching the brief's framing question: "How
can RF characteristics be used to distinguish different signal
sources?"
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

SPAN_LOW_HZ = 2_400e6
SPAN_HIGH_HZ = 2_483.5e6

WIFI_CHANNEL_CENTERS_HZ = [2_412e6, 2_437e6, 2_462e6]  # channels 1, 6, 11

ENVIRONMENTS = {
    # environment name -> baseline noise floor (dBm)
    "quiet_rural": -100.0,
    "suburban": -92.0,
    "urban": -85.0,
    "indoor_congested": -78.0,
}


@dataclass
class SignalParams:
    signal_class: str
    center_freq_hz: Optional[float]  # None for pure background
    bandwidth_hz: Optional[float]
    power_dbm: Optional[float]
    burst_duration_s: Optional[float]
    burst_period_s: Optional[float]
    environment: str
    noise_floor_dbm: float


def sample_environment(rng: random.Random) -> tuple[str, float]:
    name = rng.choice(list(ENVIRONMENTS.keys()))
    baseline = ENVIRONMENTS[name]
    # small per-sample jitter so the same environment isn't identical every time
    jitter = rng.uniform(-2.0, 2.0)
    return name, baseline + jitter


def sample_class_a_uav(rng: random.Random) -> SignalParams:
    """UAV-like: narrowband frequency-hopping bursts."""
    env_name, noise_floor = sample_environment(rng)
    return SignalParams(
        signal_class="UAV",
        center_freq_hz=rng.uniform(SPAN_LOW_HZ + 2e6, SPAN_HIGH_HZ - 2e6),
        bandwidth_hz=rng.uniform(0.5e6, 4e6),
        power_dbm=rng.uniform(-65.0, -35.0),
        burst_duration_s=rng.uniform(0.002, 0.02),
        burst_period_s=rng.uniform(0.01, 0.05),
        environment=env_name,
        noise_floor_dbm=noise_floor,
    )


def sample_class_b_wifi(rng: random.Random) -> SignalParams:
    """WiFi-like: wide channel, standard channel centers, packet bursts."""
    env_name, noise_floor = sample_environment(rng)
    center = rng.choice(WIFI_CHANNEL_CENTERS_HZ) + rng.uniform(-1e6, 1e6)
    return SignalParams(
        signal_class="WiFi",
        center_freq_hz=center,
        bandwidth_hz=rng.choice([20e6, 40e6]),
        power_dbm=rng.uniform(-75.0, -45.0),
        burst_duration_s=rng.uniform(0.05, 0.5),
        burst_period_s=rng.uniform(0.1, 1.0),
        environment=env_name,
        noise_floor_dbm=noise_floor,
    )


def sample_class_c_generic(rng: random.Random) -> SignalParams:
    """Generic narrowband comms: near-continuous carrier."""
    env_name, noise_floor = sample_environment(rng)
    duration = rng.uniform(1.0, 5.0)
    return SignalParams(
        signal_class="Generic",
        center_freq_hz=rng.uniform(SPAN_LOW_HZ + 1e6, SPAN_HIGH_HZ - 1e6),
        bandwidth_hz=rng.uniform(0.025e6, 0.5e6),
        power_dbm=rng.uniform(-80.0, -50.0),
        burst_duration_s=duration,
        burst_period_s=duration + rng.uniform(0.0, 0.5),
        environment=env_name,
        noise_floor_dbm=noise_floor,
    )


def sample_class_d_background(rng: random.Random) -> SignalParams:
    """Pure background noise - no synthesized signal."""
    env_name, noise_floor = sample_environment(rng)
    return SignalParams(
        signal_class="Background",
        center_freq_hz=None,
        bandwidth_hz=None,
        power_dbm=None,
        burst_duration_s=None,
        burst_period_s=None,
        environment=env_name,
        noise_floor_dbm=noise_floor,
    )


CLASS_SAMPLERS = {
    "UAV": sample_class_a_uav,
    "WiFi": sample_class_b_wifi,
    "Generic": sample_class_c_generic,
    "Background": sample_class_d_background,
}


def sample_signal(rng: random.Random, class_name: Optional[str] = None) -> SignalParams:
    """Draw one signal instance. If class_name is None, pick a class
    uniformly at random (used when generating a balanced dataset)."""
    if class_name is None:
        class_name = rng.choice(list(CLASS_SAMPLERS.keys()))
    return CLASS_SAMPLERS[class_name](rng)
