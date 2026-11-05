# Daphnet Freezing of Gait: leave-one-subject-out results

Seeds [0, 1, 2] · 8 test subjects with freezes (S01, S02, S03, S05, S06, S07, S08, S09) · 32 Hz · frame budget 31.2 ms · 20 epochs · hidden 32 · TCN layers 8 · 2 validation subjects per fold · input features: spectral · training noise augmentation: none

All thresholds were selected on validation subjects only. Event metrics are pooled over all test subjects within a seed (mean ± std across seeds). Sensitivity, specificity and PR-AUC are per fold (mean ± std across seeds and folds).

## Main table: online (deployable) setting

| Model | Causal | Event precision | Event recall | Event F1 | Sensitivity | Specificity | PR-AUC | Median onset delay (ms) | False alarms / min | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Freeze Index | yes | 47.5 ± 2.6 | 57.8 ± 2.8 | 52.1 ± 2.6 | 27.5 ± 20.2 | 97.4 ± 2.8 | 0.414 ± 0.180 | 1094 | 0.68 ± 0.04 | n/a (FFT, no network) |
| TCN-causal | yes | 50.1 ± 5.1 | 70.7 ± 1.8 | 58.5 ± 2.9 | 48.9 ± 31.3 | 92.5 ± 14.4 | 0.641 ± 0.148 | 250 | 0.63 ± 0.11 | 0.35 / 0.41 |

## Online with a debounced alarm (threshold and minimum alarm duration both chosen on validation subjects)

An alarm is raised once p(freeze) has stayed above the threshold for the chosen duration (0 to 1.5 s); the same rule is applied to the Freeze Index. PR-AUC of the Freeze Index is computed on the power-gated FI score that its detector uses.

| Model | Event precision | Event recall | Event F1 | False alarms / min | Median onset delay (ms) | Median debounce (s) |
|---|---|---|---|---|---|---|
| Freeze Index | 51.0 ± 3.2 | 59.6 ± 3.7 | 55.0 ± 3.2 | 0.59 ± 0.01 | 1109 | 0.38 |
| TCN-causal | 45.0 ± 0.6 | 66.9 ± 3.3 | 53.8 ± 1.3 | 0.69 ± 0.03 | 156 | 0.00 |

## Offline vs. online (event F1, pooled; mean ± std across seeds)

| Model | Offline event F1 | Online event F1 | Change | Offline PR-AUC | Online PR-AUC |
|---|---|---|---|---|---|
| TCN-causal | 58.5 ± 2.9 | 58.5 ± 2.9 | +0.0 | 0.641 ± 0.148 | 0.641 ± 0.148 |

## Per-subject online event F1 (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 38.8 | 56.2 | 52.5 | 59.6 | 28.4 | 20.1 | 37.1 | 64.0 |
| TCN-causal | 20.5 | 70.5 | 55.3 | 61.3 | 45.9 | 50.4 | 53.7 | 75.0 |

## Per-subject online recall / false alarms per minute (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 67% / 1.0 | 67% / 0.9 | 78% / 1.0 | 55% / 0.7 | 37% / 0.5 | 22% / 0.3 | 33% / 0.3 | 64% / 0.4 |
| TCN-causal | 77% / 1.7 | 62% / 0.1 | 92% / 0.5 | 55% / 0.4 | 47% / 0.5 | 78% / 0.7 | 83% / 1.1 | 69% / 0.1 |

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
| TCN-causal | 0.35 / 0.41 | 0.35 / 0.41 | 0.35 / 0.41 | 0.35 / 0.41 | 0.35 / 0.41 |

Latency measured on `x86_64`; online acausal models re-run on the last 10 s every 8 frames (250 ms), as a deployed system would have to.

Latency note: latency block copied from the daphnet_spectral_aug run (identical architecture and input size), measured on an otherwise idle CPU; this run measured latency while other jobs shared the CPU.
