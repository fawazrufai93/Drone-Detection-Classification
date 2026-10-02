"""
WEEK 1 - RF Environment & Spectrum Monitoring
Pipeline:  Simulated RF Environment -> Spectrum Scanner -> FFT/PSD
           -> Noise Estimation -> Spectrum Occupancy Map
Run:  python rf_spectrum_monitor.py      (cells marked '# %%' also work in Jupyter)
"""
# %% Imports
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # remove this line inside Jupyter to show plots inline
import matplotlib.pyplot as plt
from scipy.signal import welch, resample_poly

rng = np.random.default_rng(42)

# %% STEP 1 - Synthetic RF environment --------------------------------------
F_START, F_STOP = 2.400e9, 2.500e9      # defined frequency range (2.4 GHz ISM band)
F0 = (F_START + F_STOP) / 2             # centre of the simulated band
FS_ENV = 100e6                          # environment sample rate (>= band width, Nyquist)
NOISE_DENSITY_DBM_HZ = -160             # background noise (thermal -174 dBm/Hz + 14 dB noise figure)

# name, centre (Hz), bandwidth (Hz), power (dBm), type, probability of being ON in a dwell
SIGNALS = [
    ("WiFi ch1",        2.412e9, 20e6,  -50, "continuous",   1.0),
    ("WiFi ch11",       2.462e9, 20e6,  -60, "continuous",   1.0),
    ("Beacon (CW)",     2.428e9, 0.1e6, -65, "continuous",   1.0),
    ("Weak link",       2.485e9, 2e6,   -85, "continuous",   1.0),
    ("Bluetooth-like",  2.441e9, 1e6,   -70, "intermittent", 0.35),
    ("Wideband burst",  2.450e9, 8e6,   -55, "intermittent", 0.20),
]

def make_signal(n, bw, power_dbm):
    """Band-limited complex noise of bandwidth bw, scaled to the requested power (mW)."""
    spec = np.zeros(n, complex)
    f = np.fft.fftfreq(n, 1 / FS_ENV)
    m = np.abs(f) <= bw / 2
    spec[m] = rng.normal(size=m.sum()) + 1j * rng.normal(size=m.sum())
    x = np.fft.ifft(spec)
    return x * np.sqrt(10 ** (power_dbm / 10) / np.mean(np.abs(x) ** 2))

def rf_environment(n, active_log=None):
    """One time slice of the RF environment (complex baseband around F0)."""
    t = np.arange(n) / FS_ENV
    var = 10 ** (NOISE_DENSITY_DBM_HZ / 10) * FS_ENV            # noise power in mW
    x = np.sqrt(var / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
    for name, fc, bw, p, kind, p_on in SIGNALS:
        if rng.random() < p_on:                                 # intermittent -> random ON/OFF
            x += make_signal(n, bw, p) * np.exp(2j * np.pi * (fc - F0) * t)
            if active_log is not None:
                active_log.append(name)
    return x

# %% STEP 2 - Spectrum scanner (sweep, FFT/PSD, noise floor, occupancy, record) --
FS_RX = 20e6                     # receiver instantaneous bandwidth
STEP = 10e6                      # tuning step (keep the central 10 MHz of each capture)
DWELL = 12800                    # samples per dwell at FS_ENV  (=2560 at FS_RX)
NPERSEG = 256                    # FFT size -> frequency resolution = FS_RX/NPERSEG = 78 kHz
THRESH_DB = 6                    # occupancy threshold above noise floor

def sweep():
    """Tune across F_START..F_STOP; return frequency axis (Hz) and PSD (dBm/Hz)."""
    freqs, psd = [], []
    t = np.arange(DWELL) / FS_ENV
    for fc in np.arange(F_START + STEP / 2, F_STOP, STEP):
        x = rf_environment(DWELL)
        x = x * np.exp(-2j * np.pi * (fc - F0) * t)             # tune to fc
        x = resample_poly(x, 1, int(FS_ENV / FS_RX))            # channel filter + decimate
        f, p = welch(x, fs=FS_RX, window="hann", nperseg=NPERSEG,
                     noverlap=NPERSEG // 2, return_onesided=False, detrend=False)   # FFT -> PSD
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
        k = np.abs(f) < STEP / 2
        freqs.append(fc + f[k]); psd.append(p[k])
    return np.concatenate(freqs), 10 * np.log10(np.concatenate(psd))

def noise_floor(psd_db):
    """Robust estimate: median of the quietest half of the bins (signals occupy the rest)."""
    return np.median(np.sort(psd_db)[: len(psd_db) // 2])

def occupied_segments(freqs, psd_db, nf, thr=THRESH_DB, gap=2):
    """Group contiguous bins above nf+thr into occupied segments."""
    above = np.where(psd_db > nf + thr)[0]
    if len(above) == 0:
        return []
    groups = np.split(above, np.where(np.diff(above) > gap + 1)[0] + 1)
    out = []
    for g in groups:
        lo, hi = freqs[g[0]], freqs[g[-1]]
        out.append(dict(center_MHz=(lo + hi) / 2e6, bandwidth_MHz=(hi - lo) / 1e6 + (freqs[1] - freqs[0]) / 1e6,
                        peak_dBm_Hz=psd_db[g].max(), SNR_dB=psd_db[g].max() - nf))
    return out

def run_scanner(n_sweeps=100):
    """Repeat sweeps and RECORD the measurements."""
    records, history, floors = [], [], []
    for s in range(n_sweeps):
        freqs, psd_db = sweep()
        nf = noise_floor(psd_db)
        history.append(psd_db); floors.append(nf)
        for seg in occupied_segments(freqs, psd_db, nf):
            records.append(dict(sweep=s, noise_floor_dBm_Hz=nf, **seg))
    return freqs, np.array(history), np.array(floors), pd.DataFrame(records)

# %% STEP 3 - Run, record, and build the spectrum occupancy map ------------
if __name__ == "__main__":
    freqs, H, floors, log = run_scanner(100)
    fMHz = freqs / 1e6

    # Record the measurements
    np.savez("spectrum_measurements.npz", freqs_hz=freqs, psd_dbm_hz=H, noise_floor=floors)
    log.to_csv("occupied_segments_log.csv", index=False)

    # Occupancy map = fraction of sweeps in which each frequency bin is above nf + threshold
    occ = (H > (floors[:, None] + THRESH_DB)).mean(axis=0) * 100

    # Plot 1: a single spectrum measurement with noise floor + occupied regions
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(fMHz, H[0], lw=0.8, label="PSD (sweep 0)")
    ax.axhline(floors[0], color="g", ls="--", label=f"Noise floor {floors[0]:.1f} dBm/Hz")
    ax.axhline(floors[0] + THRESH_DB, color="r", ls=":", label=f"Threshold (+{THRESH_DB} dB)")
    for seg in occupied_segments(freqs, H[0], floors[0]):
        ax.axvspan(seg["center_MHz"] - seg["bandwidth_MHz"] / 2,
                   seg["center_MHz"] + seg["bandwidth_MHz"] / 2, color="orange", alpha=0.25)
    ax.set(xlabel="Frequency (MHz)", ylabel="PSD (dBm/Hz)", title="Spectrum measurement (FFT/PSD)")
    ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig("1_spectrum_psd.png", dpi=150)

    # Plot 2: waterfall (measurement history)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis",
                   extent=[fMHz[0], fMHz[-1], 0, len(H)])
    fig.colorbar(im, label="dBm/Hz"); ax.set(xlabel="Frequency (MHz)", ylabel="Sweep #", title="Waterfall")
    fig.tight_layout(); fig.savefig("2_waterfall.png", dpi=150)

    # Plot 3: spectrum occupancy map
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(fMHz, occ, color="tab:red", alpha=0.6)
    ax.set(xlabel="Frequency (MHz)", ylabel="Occupancy (% of sweeps)", ylim=(0, 105),
           title="Spectrum Occupancy Map")
    fig.tight_layout(); fig.savefig("3_occupancy_map.png", dpi=150)

    # Summary of the detected signals (averaged over all sweeps)
    log["band"] = (log["center_MHz"] / 2).round() * 2
    print(f"True noise floor: {NOISE_DENSITY_DBM_HZ} dBm/Hz | estimated (mean of sweeps): {floors.mean():.2f} dBm/Hz")
    print(f"Frequency resolution: {FS_RX / NPERSEG / 1e3:.1f} kHz | sweeps recorded: {len(H)}")
    print("\nDetected segments (grouped by centre frequency):")
    print(log.groupby("band").agg(seen_in_sweeps=("sweep", "nunique"), bw_MHz=("bandwidth_MHz", "mean"),
                                  peak=("peak_dBm_Hz", "mean"), SNR_dB=("SNR_dB", "mean")).round(2))
