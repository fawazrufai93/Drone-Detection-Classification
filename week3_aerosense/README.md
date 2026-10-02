# Week 3 - Drone Detection & Classification: AeroSense RF Detection Engine v1

Built from the Week 3 brief, using the **Week 2 synthetic dataset / signal pipeline** and the **Week 1 spectrum-monitor
functions** (copied unchanged into `inputs/`).

```
RF Activity Detected -> Signal Features -> Classification -> Confidence -> Alert
```


```

| File | Role |
|---|---|
| `aerosense/detector.py` | Task 1: RF input -> noise estimation -> threshold -> signal detected? (Week 1 `noise_floor`, `occupied_segments`) |
| `aerosense/classifier.py`, `task2_classification.py` | Task 2: decision tree / random forest / SVM / k-NN on the Week 2 dataset |
| `aerosense/engine.py` | Deliverable: detection -> features (Week 2 extractor) -> classification -> confidence -> alert |
| `aerosense/simulate.py` | Test-signal generator (Week 2 profiles + IQ model; adds SNR control and multi-signal placement) |
| `task3_false_alarm.py` | Task 3: five stress scenarios |

## Task 1 - Detection (tested at different SNR)
Noise floor = Week 1 estimator (median of the quietest half of the PSD bins). Threshold = noise floor + **8 dB**.
Bins above it are grouped into occupied segments; any segment = "signal detected".

Detection probability vs in-band SNR (200 trials per cell, 0 false alarms in 2000 noise-only snapshots):

| SNR (dB) | -6 | -3 | 0 | 3 | 6 | 9 | 12+ |
|---|---|---|---|---|---|---|---|
| UAV | 0.00 | 0.00 | 0.02 | 0.45 | 1.00 | 1.00 | 1.00 |
| Wi-Fi | 0.00 | 0.00 | 0.01 | 0.15 | 0.96 | 1.00 | 1.00 |
| Generic | 0.00 | 0.00 | 0.01 | 0.14 | 0.82 | 0.98 | 1.00 |

Validated on the Week 1 recorded sweeps (100 sweeps, Week 1's own 6 dB setting): all four always-on sources found in 100% of
sweeps, the intermittent ones in 42% / 41% of sweeps, no segment unmatched to a known source.

**Threshold choice.** Week 1's 6 dB margin gives a **53%** noise-only false alarm rate on a Week 2 snapshot (only 7 averaged
Welch segments, versus ~19 in Week 1). Measured false alarm rate vs margin: 6.5 dB 15% - 7 dB 2.8% - 7.5 dB 0.4% - **8 dB 0 / 2000**.
(`results/task1_margin_calibration.csv`, `task1_roc.png`.)

## Task 2 - Classification (Week 2 dataset, 5000 rows, 75/25 split, 6 measured spectral features)
Features: frequency, bandwidth, power, SNR, spectral flatness, PAPR.

| Model | 5-fold CV | Test accuracy |
|---|---|---|
| Decision tree | 0.996 | 0.996 |
| **Random forest** (used) | **0.997** | **0.998** |
| SVM | 0.966 | 0.972 |
| k-NN | 0.976 | 0.972 |

Feature importance: bandwidth 0.46, flatness 0.16, power/PAPR/SNR ~0.13 each, frequency 0.005. Confusion matrices in
`results/task2_confusion_matrices.png`.

## Task 3 - False alarm analysis (full engine, 300 captures per scenario + 300 noise-only at the same noise floor)

| Scenario | Detection rate (detector) | Detection rate (engine) | False alarm rate (noise only) | Classification accuracy |
|---|---|---|---|---|
| Low noise (-100 dBm) | 100% | 100% | 0% | 100% |
| High noise (-70 dBm) | 94% | 94% | 0% | 97.5% |
| Multiple signals (3) | 95.7% | 95.2% | 0% | 83.1% |
| Overlapping signals (2) | 99.8% | 99.8% | 0% | 47.4% |
| Weak signals (0-10 dB SNR) | 66.7% | **8.7%** | 0% | 42.3% |

Where it breaks:
- **Overlapping signals** - bands merge into one segment, so one label is given to two signals (accuracy 47%).
- **Weak signals** - the detector still sees 2/3 of them, but the classifier rejects most as "Background": the Week 2 dataset only
  contains strong signals (in-band SNR ~35-70 dB), so low power/SNR values are outside what it was trained on.
- **Multiple signals** - Wi-Fi is missed when wideband signals fill >50% of the band (noise-floor estimate is contaminated); narrow
  signals sitting on a strong neighbour's skirts get the wrong bandwidth.
- Remaining false *detections* (0.06-0.20 per capture) are skirt/band-edge fragments of real signals, not noise (noise-only alarms are 0%).

## Engine example (`results/engine_demo.txt`)
```
RF Activity Detected : YES  (signal(s); ...)
  [1] Signal Features : f=2470.16 MHz  BW=4.35 MHz  P=-18.4  SNR=72.5 dB ...
      Classification : UAV
      Confidence     : 1.00
      Alert          : *** UAV ALERT *** at 2470.16 MHz
```
Alert rule: class = UAV and confidence (random-forest probability) >= 0.70.

## Decisions and caveats
1. **Detector uses its own PSD, not Week 2's `compute_psd`.** That function adds `+1e-20` before the log; at noise floors of
   -90 dBm and below this is as large as the noise PSD and hides its variance, making noise-only false alarms look like 0%.
   Same Welch settings and calibration otherwise. Week 2 files are unmodified and still used for feature extraction.
2. **Classifier uses the 6 spectral features only.** `duration` / `burst_interval` need a ~2 s envelope; the engine's input is a
   single IQ snapshot. (Using all 8 features gives 99.9% on the dataset, vs 99.8% here.)
3. **Segments merged within 1 MHz** (Week 1 used 2 bins): without it, noisy skirts and wideband ripple fragment one signal into
   many "detections" (up to ~30 per Wi-Fi capture).
4. **Bandwidth for features = 90%-power bandwidth**, not threshold width: threshold width grows with SNR and made strong UAV
   signals look like Wi-Fi (low-noise accuracy 88% -> 100%). The 90% value was picked on a separate 300-capture test.
5. Segments the classifier labels "Background" are rejected as not-a-signal.
6. Wi-Fi ch1 at 40 MHz extends past the 84 MHz analysis window in the Week 2 profiles and wraps to the other edge; Task 3 scoring
   treats that wrapped copy as part of the signal.
7. Dataset `snr`/`power` are the Week 2 definitions (integrated occupied power vs. a PSD floor), so they read higher than in-band SNR.
