"""
generate_dataset.py

Generates the Week 2 deliverable: "RF Signature Dataset v1" - several
thousand synthetically generated RF signal samples across four classes
(UAV, WiFi, Generic, Background), each run through the full
Synthetic Signal -> Noise -> FFT -> Filtering -> Feature Extraction
pipeline, with the result written out as a CSV dataset.

Usage:
    python3 generate_dataset.py [n_samples] [output_path]

Defaults to 5000 samples, written to rf_signature_dataset_v1.csv in
the current directory. Classes are balanced (roughly n_samples/4 of
each), and each sample gets a synthetic, monotonically increasing
timestamp to simulate a detection log collected over time.
"""

from __future__ import annotations

import csv
import random
import sys
from datetime import datetime, timedelta

import numpy as np

from pipeline import build_feature_row
from signal_profiles import CLASS_SAMPLERS, sample_signal

FIELDNAMES = [
    "timestamp",
    "frequency",
    "bandwidth",
    "power",
    "snr",
    "duration",
    "signal_type",
    "environment",
    "burst_interval",
    "spectral_flatness",
    "papr_db",
    "noise_floor_dbm",
]


def generate_dataset(n_samples: int, output_path: str, seed: int = 42) -> None:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    class_names = list(CLASS_SAMPLERS.keys())
    # Balanced class assignment: repeat the class list enough times to
    # cover n_samples, then shuffle so the file isn't grouped by class.
    assignments = (class_names * (n_samples // len(class_names) + 1))[:n_samples]
    py_rng.shuffle(assignments)

    current_time = datetime(2026, 9, 24, 6, 0, 0)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for i, class_name in enumerate(assignments):
            params = sample_signal(py_rng, class_name=class_name)
            row = build_feature_row(params, np_rng, py_rng)

            # Simulate a detection log: gaps between consecutive log
            # entries drawn from an exponential distribution (typical
            # for event arrival processes), mean 0.8s.
            current_time += timedelta(seconds=py_rng.expovariate(1 / 0.8))

            writer.writerow(
                {
                    "timestamp": current_time.isoformat(timespec="milliseconds"),
                    "frequency": row["frequency"],
                    "bandwidth": row["bandwidth"],
                    "power": row["power"],
                    "snr": row["snr"],
                    "duration": row["duration"],
                    "signal_type": row["signal_type"],
                    "environment": row["environment"],
                    "burst_interval": row["burst_interval"],
                    "spectral_flatness": row["spectral_flatness"],
                    "papr_db": row["papr_db"],
                    "noise_floor_dbm": row["noise_floor_dbm"],
                }
            )

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{n_samples} samples generated...")

    print(f"Done. Wrote {n_samples} samples to {output_path}")


if __name__ == "__main__":
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    output_path = sys.argv[2] if len(sys.argv) > 2 else "rf_signature_dataset_v1.csv"
    print(f"Generating {n_samples} samples -> {output_path}")
    generate_dataset(n_samples, output_path)
