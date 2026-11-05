# Daphnet Freezing of Gait: leave-one-subject-out results

Seeds [0, 1, 2] · 8 test subjects with freezes (S01, S02, S03, S05, S06, S07, S08, S09) · 32 Hz · frame budget 31.2 ms · 20 epochs · hidden 64 · TCN layers 10 · 2 validation subjects per fold · input features: spectral · training noise augmentation: none

All thresholds were selected on validation subjects only. Event metrics are pooled over all test subjects within a seed (mean ± std across seeds). Sensitivity, specificity and PR-AUC are per fold (mean ± std across seeds and folds).

## Main table: online (deployable) setting

| Model | Causal | Event precision | Event recall | Event F1 | Sensitivity | Specificity | PR-AUC | Median onset delay (ms) | False alarms / min | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Freeze Index | yes | 47.5 ± 2.6 | 57.8 ± 2.8 | 52.1 ± 2.6 | 27.5 ± 20.2 | 97.4 ± 2.8 | 0.414 ± 0.180 | 1094 | 0.68 ± 0.04 | n/a (FFT, no network) |
| TCN-causal | yes | 53.1 ± 2.1 | 72.3 ± 1.4 | 61.2 ± 1.8 | 46.2 ± 30.9 | 91.8 ± 14.6 | 0.599 ± 0.166 | 203 | 0.61 ± 0.04 | 1.02 / 1.51 |

## Online with a debounced alarm (threshold and minimum alarm duration both chosen on validation subjects)

An alarm is raised once p(freeze) has stayed above the threshold for the chosen duration (0 to 1.5 s); the same rule is applied to the Freeze Index. PR-AUC of the Freeze Index is computed on the power-gated FI score that its detector uses.

| Model | Event precision | Event recall | Event F1 | False alarms / min | Median onset delay (ms) | Median debounce (s) |
|---|---|---|---|---|---|---|
| Freeze Index | 51.0 ± 3.2 | 59.6 ± 3.7 | 55.0 ± 3.2 | 0.59 ± 0.01 | 1109 | 0.38 |
| TCN-causal | 47.4 ± 3.2 | 67.4 ± 5.3 | 55.4 ± 1.1 | 0.72 ± 0.14 | 312 | 0.25 |

## Offline vs. online (event F1, pooled; mean ± std across seeds)

| Model | Offline event F1 | Online event F1 | Change | Offline PR-AUC | Online PR-AUC |
|---|---|---|---|---|---|
| TCN-causal | 61.2 ± 1.8 | 61.2 ± 1.8 | +0.0 | 0.599 ± 0.166 | 0.599 ± 0.166 |

## Per-subject online event F1 (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 38.8 | 56.2 | 52.5 | 59.6 | 28.4 | 20.1 | 37.1 | 64.0 |
| TCN-causal | 22.8 | 68.2 | 61.3 | 64.7 | 42.8 | 45.9 | 50.0 | 76.1 |

## Per-subject online recall / false alarms per minute (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 67% / 1.0 | 67% / 0.9 | 78% / 1.0 | 55% / 0.7 | 37% / 0.5 | 22% / 0.3 | 33% / 0.3 | 64% / 0.4 |
| TCN-causal | 75% / 1.1 | 57% / 0.1 | 90% / 0.5 | 59% / 0.6 | 37% / 0.2 | 90% / 1.1 | 81% / 1.3 | 75% / 0.3 |

## Class balance of the test subjects

| Subject | Freeze frames (%) | Freeze episodes | Test minutes |
|---|---|---|---|
| S01 | 5.5 | 23 | 31.7 |
| S02 | 12.8 | 24 | 23.6 |
| S03 | 14.3 | 49 | 33.5 |
| S05 | 22.7 | 66 | 34.8 |
| S06 | 6.6 | 10 | 33.2 |
| S07 | 5.1 | 24 | 26.8 |
| S08 | 26.1 | 14 | 12.8 |
| S09 | 15.5 | 27 | 29.0 |
| **All** | 12.8 | 237 | 225.4 |

## Per-update latency (CPU, batch size 1, median / p95 ms) by elapsed history

| Model | 10 s | 30 s | 60 s | 120 s | 300 s |
|---|---|---|---|---|---|
| TCN-causal | 1.02 / 1.51 | 1.02 / 1.51 | 1.02 / 1.51 | 1.02 / 1.51 | 1.02 / 1.51 |

Latency measured on `x86_64`; online acausal models re-run on the last 10 s every 8 frames (250 ms), as a deployed system would have to.
