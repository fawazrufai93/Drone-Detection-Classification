"""
aerosense_engine.py  --  Week 3 Deliverable: AeroSense RF Detection Engine v1

    RF Activity Detected -> Signal Features -> Classification -> Confidence -> Alert

Built from the earlier tasks:
    detector.py   (Task 1)  -> RF activity detected?
    pipeline.py   (Week 2)  -> signal features from the spectrum
    classifier    (Task 2)  -> random forest trained on rf_signature_dataset_v1.csv

Training: the classifier is trained on signals generated from the same Week 2
signal profiles (signal_profiles.py) but with features measured by THIS
engine's own extraction path, so training and live features match. (The
Week 2 CSV features were measured with a filter centred on the true signal,
which a live engine does not know.) The training set is cached in
engine_training_data.csv.

Note: only spectral features are used (frequency, bandwidth, power, SNR,
spectral flatness, PAPR, noise floor). Burst duration / interval need a
much longer observation than one 0.2 ms IQ sweep, so they are left out.

Usage:
    python3 aerosense_engine.py
"""

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import pipeline as pl
from detector import SignalDetector
from signal_profiles import sample_signal

SPECTRAL_FEATURES = ["frequency", "bandwidth", "power", "snr",
                     "spectral_flatness", "papr_db", "noise_floor_dbm"]

TRAIN_CSV = "engine_training_data.csv"
ALERT_CONFIDENCE = 0.80      # UAV label needs at least this confidence for a full alert


@dataclass
class EngineResult:
    activity_detected: bool
    features: Optional[dict]
    label: Optional[str]
    confidence: Optional[float]
    alert: str

    def summary(self) -> str:
        if not self.activity_detected:
            return "No RF activity"
        f = self.features
        return (f"{f['frequency']:.1f} MHz, BW {f['bandwidth']:.2f} MHz, SNR {f['snr']:.1f} dB "
                f"-> {self.label} ({self.confidence:.2f})")


class AeroSenseEngine:
    def __init__(self, margin_db: float = 3.0, min_bins: int = 3) -> None:
        self.detector = SignalDetector(margin_db, min_bins)
        data = self._training_data()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(data[SPECTRAL_FEATURES], data["signal_type"])

    def _training_data(self, n_per_class: int = 500, seed: int = 11) -> pd.DataFrame:
        if os.path.exists(TRAIN_CSV):
            return pd.read_csv(TRAIN_CSV)
        print(f"Building training set ({4 * n_per_class} samples, first run only)...")
        rng, py = np.random.default_rng(seed), random.Random(seed)
        rows = []
        for cls in ["UAV", "WiFi", "Generic", "Background"]:
            for _ in range(n_per_class):
                p = sample_signal(py, class_name=cls)
                iq = _synth(p, rng)
                nf = self.detector.detect(iq).noise_floor_dbm
                rows.append({**self.extract_features(iq, nf), "signal_type": cls})
        df = pd.DataFrame(rows)
        df.to_csv(TRAIN_CSV, index=False)
        return df

    # ---- stage 2: signal features -------------------------------------
    def extract_features(self, iq: np.ndarray, noise_floor_dbm: float) -> dict:
        f, pw = pl.compute_psd(iq)
        floor = float(np.percentile(pw, 15))
        raw = pw > floor + 6.0
        if raw.any():
            run = pl._largest_contiguous_run(raw, 10 ** (pw / 10))     # occupied region
            lin = 10 ** (pw[run] / 10)
            c_mhz = float(np.sum(f[run] * lin) / np.sum(lin))
            bw_mhz = float(f[run].max() - f[run].min())
            iq = pl.bandpass_filter(iq, c_mhz * 1e6, bw_mhz * 1e6)      # filtering stage
            f, pw = pl.compute_psd(iq)
        s = pl.extract_spectral_features(f, pw, iq)
        return {
            "frequency": s["center_freq_mhz"] or 0.0,
            "bandwidth": s["bandwidth_mhz"] or 0.0,
            "power": s["power_dbm"], "snr": s["snr_db"],
            "spectral_flatness": s["spectral_flatness"], "papr_db": s["papr_db"],
            "noise_floor_dbm": noise_floor_dbm,
        }

    # ---- stage 5: alert ------------------------------------------------
    @staticmethod
    def make_alert(label: str, confidence: float) -> str:
        if label == "UAV":
            if confidence >= ALERT_CONFIDENCE:
                return "ALERT: possible UAV activity"
            return "REVIEW: UAV-like signal, low confidence"
        if label == "Background":
            return "REVIEW: RF activity detected but not classifiable"
        return f"INFO: {label} signal (not UAV-like)"

    # ---- full chain ----------------------------------------------------
    def process(self, iq: np.ndarray) -> EngineResult:
        det = self.detector.detect(iq)                                   # 1. RF activity detected?
        if not det.detected:
            return EngineResult(False, None, None, None, "NONE: no RF activity")
        feats = self.extract_features(iq, det.noise_floor_dbm)           # 2. signal features
        row = pd.DataFrame([feats], columns=SPECTRAL_FEATURES)
        proba = self.classifier.predict_proba(row)[0]                    # 3. classification
        label = str(self.classifier.classes_[int(np.argmax(proba))])
        confidence = float(proba.max())                                  # 4. confidence
        return EngineResult(True, feats, label, confidence,              # 5. alert
                            self.make_alert(label, confidence))


# ---------------------------------------------------------------- demo
def _synth(params, rng):
    n, fs = 16384, pl.ANALYSIS_SAMPLE_RATE_HZ
    t = np.arange(n) / fs
    a = pl.dbm_to_amplitude(params.noise_floor_dbm)
    iq = a / np.sqrt(2) * (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n))
    if params.center_freq_hz is not None:
        sh = pl._generate_bandlimited_noise(n, params.bandwidth_hz, rng)
        iq = iq + pl.dbm_to_amplitude(params.power_dbm) * sh * np.exp(
            2j * np.pi * (params.center_freq_hz - pl.ANALYSIS_CENTER_FREQ_HZ) * t)
    return iq


def main(seed: int = 3, n_eval: int = 150) -> None:
    engine = AeroSenseEngine()
    rng = np.random.default_rng(seed)
    py = random.Random(seed)

    print("AeroSense RF Detection Engine v1")
    print("RF Activity Detected -> Signal Features -> Classification -> Confidence -> Alert\n")
    print(f"{'Truth':<11}{'Detected':<10}{'Class':<12}{'Conf.':<7}Alert")
    print("-" * 78)
    for truth in ["Background", "UAV", "WiFi", "Generic", "UAV", "WiFi", "Generic", "Background"]:
        p = sample_signal(py, class_name=truth)
        r = engine.process(_synth(p, rng))
        print(f"{truth:<11}{'Yes' if r.activity_detected else 'No':<10}"
              f"{(r.label or '-'):<12}{(f'{r.confidence:.2f}' if r.confidence else '-'):<7}{r.alert}")

    print(f"\nEvaluation: {n_eval} trials per true signal type (random environment noise)")
    print(f"{'Truth':<11}{'Full UAV alert':>16}{'UAV-like (any)':>17}")
    print("-" * 44)
    for truth in ["UAV", "WiFi", "Generic", "Background"]:
        full = any_uav = 0
        for _ in range(n_eval):
            r = engine.process(_synth(sample_signal(py, class_name=truth), rng))
            full += r.alert.startswith("ALERT")
            any_uav += (r.label == "UAV")
        print(f"{truth:<11}{full / n_eval:>16.3f}{any_uav / n_eval:>17.3f}")
    print("\n'Full UAV alert' = UAV label with confidence >= "
          f"{ALERT_CONFIDENCE:.2f}.  For non-UAV rows this is the false-alarm rate.")


if __name__ == "__main__":
    main()
