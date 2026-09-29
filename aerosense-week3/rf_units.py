"""
rf_units.py

Shared power/amplitude conversion so the simulator (which injects
signals at a given dBm) and the analyzer (which measures dBm back out
via Welch's method) agree on the same reference scale.
"""

import numpy as np


def dbm_to_amplitude(power_dbm: float) -> float:
    """Convert a dBm power figure into an equivalent IQ sample amplitude
    on a nominal 1-ohm, 1 mW = 0 dBm reference scale."""
    power_mw = 10 ** (power_dbm / 10)
    return float(np.sqrt(power_mw / 1000))
