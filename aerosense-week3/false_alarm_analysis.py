"""
false_alarm_analysis.py  --  Week 3, Task 3: False Alarm Analysis

Scenarios tested:
    low noise | high noise | multiple signals | overlapping signals | weak signals

Measured for each scenario:
    detection rate, false alarm rate, classification accuracy

Chain used for every trial (Task 1 + Task 2 re-used):
    IQ -> SignalDetector (detector.py)            -> detected?  yes/no
       -> feature extraction (Week 2 pipeline.py) -> feature vector
       -> Random forest (from classifier.py data)   -> signal class

Definitions
    detection rate       = detected / trials that contain a signal
    false alarm rate     = flagged  / noise-only trials
    classification acc.  = correct class / signal trials that were detected
                           (truth = class of the strongest signal present)
"""

import random
import numpy as np

import pipeline as pl
from signal_profiles import sample_signal, WIFI_CHANNEL_CENTERS_HZ
from detector import SignalDetector
from sklearn.ensemble import RandomForestClassifier
from classifier import load_data, FEATURES

TRIALS = 200          # signal trials AND noise-only trials per scenario
FS = pl.ANALYSIS_SAMPLE_RATE_HZ
N = 16384             # same sweep length as detector.py
CENTER = pl.ANALYSIS_CENTER_FREQ_HZ


# ----------------------------------------------------------- IQ synthesis
def synthesize(signals, noise_floor_dbm, rng):
    """Noise floor + any number of band-limited signals (SignalParams list)."""
    t = np.arange(N) / FS
    amp_n = pl.dbm_to_amplitude(noise_floor_dbm)
    iq = amp_n / np.sqrt(2) * (rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N))
    for s in signals:
        shaped = pl._generate_bandlimited_noise(N, s.bandwidth_hz, rng)
        carrier = np.exp(2j * np.pi * (s.center_freq_hz - CENTER) * t)
        iq = iq + pl.dbm_to_amplitude(s.power_dbm) * shaped * carrier
    return iq


def draw(cls, noise_floor, py_rng):
    p = sample_signal(py_rng, class_name=cls)
    p.noise_floor_dbm = noise_floor
    return p


def overlaps(a, b):
    return abs(a.center_freq_hz - b.center_freq_hz) < (a.bandwidth_hz + b.bandwidth_hz) / 2


# -------------------------------------------------------------- scenarios
SIGNAL_CLASSES = ["UAV", "WiFi", "Generic"]


def sc_single(noise_floor):
    def make(py_rng):
        return [draw(py_rng.choice(SIGNAL_CLASSES), noise_floor, py_rng)]
    return noise_floor, make


def sc_multiple(py_rng, noise_floor=-92.0):
    while True:
        sigs = [draw(c, noise_floor, py_rng) for c in SIGNAL_CLASSES]
        if not any(overlaps(a, b) for i, a in enumerate(sigs) for b in sigs[i + 1:]):
            return sigs


def sc_overlapping(py_rng, noise_floor=-92.0):
    a_cls, b_cls = py_rng.choice([("WiFi", "UAV"), ("WiFi", "Generic"), ("UAV", "Generic")])
    a, b = draw(a_cls, noise_floor, py_rng), draw(b_cls, noise_floor, py_rng)
    # put the second signal inside the first one's band
    b.center_freq_hz = a.center_freq_hz + py_rng.uniform(-0.4, 0.4) * a.bandwidth_hz
    return [a, b]


def sc_weak(py_rng, noise_floor=-92.0):
    p = draw(py_rng.choice(SIGNAL_CLASSES), noise_floor, py_rng)
    snr_bin = py_rng.uniform(0.0, 8.0)                       # per-bin SNR, as in Task 1
    p.power_dbm = noise_floor + snr_bin - 10 * np.log10(FS / p.bandwidth_hz)
    return [p]


SCENARIOS = {
    "Low noise":           (-100.0, lambda r: sc_single(-100.0)[1](r)),
    "High noise":          (-78.0,  lambda r: sc_single(-78.0)[1](r)),
    "Multiple signals":    (-92.0,  sc_multiple),
    "Overlapping signals": (-92.0,  sc_overlapping),
    "Weak signals":        (-92.0,  sc_weak),
}


# ---------------------------------------------------- feature extraction
def total_power_dbm(p):
    return p.power_dbm


def extract_features(iq, primary, noise_est_dbm, py_rng):
    """Week 2 chain: PSD -> occupied region -> bandpass filter -> features."""
    f, pw = pl.compute_psd(iq)
    floor = float(np.percentile(pw, 15))
    raw = pw > floor + 6.0
    if raw.any():
        run = pl._largest_contiguous_run(raw, 10 ** (pw / 10))
        lin = 10 ** (pw[run] / 10)
        c_mhz = float(np.sum(f[run] * lin) / np.sum(lin))
        bw_mhz = float(f[run].max() - f[run].min())
        filt = pl.bandpass_filter(iq, c_mhz * 1e6, bw_mhz * 1e6)
        f2, p2 = pl.compute_psd(filt)
        spec = pl.extract_spectral_features(f2, p2, filt)
    else:
        spec = pl.extract_spectral_features(f, pw, iq)
    burst = pl.extract_burst_features(primary, py_rng)   # timing branch (not in a single sweep)
    row = {
        "frequency": spec["center_freq_mhz"], "bandwidth": spec["bandwidth_mhz"],
        "power": spec["power_dbm"], "snr": spec["snr_db"],
        "duration": burst["duration_s"], "burst_interval": burst["burst_interval_s"],
        "spectral_flatness": spec["spectral_flatness"], "papr_db": spec["papr_db"],
        "noise_floor_dbm": noise_est_dbm,
    }
    return np.array([[row[k] if row[k] is not None else 0.0 for k in FEATURES]])


# -------------------------------------------------------------------- run
def run(seed=7, trials=TRIALS, verbose=True):
    X, y = load_data()
    clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X.values, y.values)
    det = SignalDetector(margin_db=3.0, min_bins=3)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    results = []

    for name, (noise_floor, make) in SCENARIOS.items():
        # --- noise-only trials -> false alarm rate
        false_alarms = 0
        for _ in range(trials):
            iq = synthesize([], noise_floor, rng)
            false_alarms += det.detect(iq).detected

        # --- signal trials -> detection rate, classification accuracy
        detected = correct = 0
        for _ in range(trials):
            sigs = make(py_rng)
            iq = synthesize(sigs, noise_floor, rng)
            res = det.detect(iq)
            if not res.detected:
                continue
            detected += 1
            truth = max(sigs, key=total_power_dbm).signal_class
            feats = extract_features(iq, max(sigs, key=total_power_dbm), res.noise_floor_dbm, py_rng)
            pred = clf.predict(feats)[0]
            correct += (pred == truth)

        results.append((name, noise_floor, detected / trials, false_alarms / trials,
                        correct / detected if detected else float("nan")))

    if verbose:
        print(f"Detector: noise floor + 3.0 dB, min_bins = 3   |   {trials} signal + {trials} noise-only trials per scenario\n")
        print(f"{'Scenario':<21}{'Noise (dBm)':>12}{'Detection rate':>16}{'False alarm rate':>18}{'Class. accuracy':>17}")
        print("-" * 84)
        for name, nf, dr, fa, acc in results:
            print(f"{name:<21}{nf:>12.0f}{dr:>16.3f}{fa:>18.3f}{acc:>17.3f}")
    return results


if __name__ == "__main__":
    run()
