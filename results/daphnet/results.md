# Daphnet Freezing of Gait: leave-one-subject-out results

Seeds [0, 1, 2] · 8 test subjects with freezes (S01, S02, S03, S05, S06, S07, S08, S09) · 32 Hz · frame budget 31.2 ms · 12 epochs · hidden 32 · TCN layers 6 · 2 validation subjects per fold

All thresholds were selected on validation subjects only. Event metrics are pooled over all test subjects within a seed (mean ± std across seeds). Sensitivity, specificity and PR-AUC are per fold (mean ± std across seeds and folds).

## Main table: online (deployable) setting

| Model | Causal | Event precision | Event recall | Event F1 | Sensitivity | Specificity | PR-AUC | Median onset delay (ms) | False alarms / min | Latency median / p95 (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Freeze Index | yes | 47.5 ± 2.6 | 57.8 ± 2.8 | 52.1 ± 2.6 | 27.5 ± 20.2 | 97.4 ± 2.8 | 0.414 ± 0.180 | 1094 | 0.68 ± 0.04 | n/a (FFT, no network) |
| TCN-causal | yes | 38.3 ± 3.4 | 65.5 ± 8.4 | 48.1 ± 3.9 | 36.3 ± 33.3 | 92.8 ± 13.4 | 0.528 ± 0.167 | 250 | 1.11 ± 0.07 | 0.49 / 0.73 |
| TCN-acausal | no | 39.9 ± 8.3 | 61.0 ± 6.2 | 47.5 ± 5.9 | 43.3 ± 33.0 | 89.8 ± 16.5 | 0.520 ± 0.177 | 359 | 1.00 ± 0.18 | 3.08 / 3.42 |
| LSTM | yes | 17.6 ± 2.3 | 57.0 ± 2.1 | 26.7 ± 2.6 | 48.2 ± 37.1 | 81.8 ± 17.9 | 0.376 ± 0.186 | 0 | 1.63 ± 0.16 | 0.16 / 0.26 |
| BiLSTM | no | 25.2 ± 2.4 | 61.6 ± 6.8 | 35.7 ± 3.4 | 46.6 ± 33.8 | 81.6 ± 18.3 | 0.363 ± 0.195 | 156 | 1.60 ± 0.19 | 17.40 / 21.23 |

## Online with a debounced alarm (threshold and minimum alarm duration both chosen on validation subjects)

An alarm is raised once p(freeze) has stayed above the threshold for the chosen duration (0 to 1.5 s); the same rule is applied to the Freeze Index. PR-AUC of the Freeze Index is computed on the power-gated FI score that its detector uses.

| Model | Event precision | Event recall | Event F1 | False alarms / min | Median onset delay (ms) | Median debounce (s) |
|---|---|---|---|---|---|---|
| Freeze Index | 51.0 ± 3.2 | 59.6 ± 3.7 | 55.0 ± 3.2 | 0.59 ± 0.01 | 1109 | 0.38 |
| TCN-causal | 35.4 ± 7.7 | 64.8 ± 6.5 | 45.1 ± 6.7 | 1.11 ± 0.09 | 219 | 0.00 |
| TCN-acausal | 39.9 ± 7.0 | 58.6 ± 4.3 | 47.0 ± 5.0 | 0.97 ± 0.15 | 688 | 0.25 |
| LSTM | 16.4 ± 3.8 | 58.4 ± 5.4 | 25.4 ± 4.8 | 1.75 ± 0.47 | 0 | 0.25 |
| BiLSTM | 22.5 ± 5.7 | 56.7 ± 9.7 | 31.3 ± 5.1 | 1.71 ± 0.73 | 0 | 1.00 |

## Offline vs. online (event F1, pooled; mean ± std across seeds)

| Model | Offline event F1 | Online event F1 | Change | Offline PR-AUC | Online PR-AUC |
|---|---|---|---|---|---|
| TCN-causal | 48.1 ± 3.9 | 48.1 ± 3.9 | +0.0 | 0.528 ± 0.167 | 0.528 ± 0.167 |
| TCN-acausal | 48.0 ± 4.7 | 47.5 ± 5.9 | -0.5 | 0.537 ± 0.173 | 0.520 ± 0.177 |
| LSTM | 26.7 ± 2.6 | 26.7 ± 2.6 | +0.0 | 0.376 ± 0.186 | 0.376 ± 0.186 |
| BiLSTM | 31.8 ± 0.9 | 35.7 ± 3.4 | +4.0 | 0.383 ± 0.205 | 0.363 ± 0.195 |

## Per-subject online event F1 (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 38.8 | 56.2 | 52.5 | 59.6 | 28.4 | 20.1 | 37.1 | 64.0 |
| TCN-causal | 14.3 | 70.4 | 62.4 | 58.6 | 27.7 | 29.5 | 38.3 | 50.0 |
| TCN-acausal | 19.5 | 72.8 | 57.8 | 52.3 | 23.5 | 22.8 | 34.2 | 50.6 |
| LSTM | 9.9 | 51.6 | 28.4 | 27.4 | 11.2 | 12.0 | 37.8 | 29.5 |
| BiLSTM | 7.8 | 67.0 | 50.1 | 40.7 | 16.0 | 13.3 | 37.9 | 36.6 |

## Per-subject online recall / false alarms per minute (mean over seeds)

| Model | S01 | S02 | S03 | S05 | S06 | S07 | S08 | S09 |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | 67% / 1.0 | 67% / 0.9 | 78% / 1.0 | 55% / 0.7 | 37% / 0.5 | 22% / 0.3 | 33% / 0.3 | 64% / 0.4 |
| TCN-causal | 88% / 2.2 | 93% / 0.7 | 62% / 0.6 | 49% / 0.5 | 33% / 0.7 | 90% / 2.6 | 64% / 1.3 | 58% / 0.5 |
| TCN-acausal | 84% / 1.2 | 89% / 0.6 | 54% / 0.5 | 43% / 0.5 | 40% / 0.9 | 85% / 2.8 | 60% / 1.5 | 59% / 0.6 |
| LSTM | 83% / 3.8 | 68% / 1.6 | 69% / 0.7 | 19% / 0.3 | 87% / 1.2 | 86% / 3.3 | 48% / 1.2 | 63% / 1.2 |
| BiLSTM | 100% / 3.9 | 76% / 0.7 | 63% / 1.0 | 29% / 0.4 | 87% / 1.8 | 86% / 2.7 | 69% / 1.7 | 57% / 0.7 |

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
| TCN-causal | 0.49 / 0.73 | 0.49 / 0.73 | 0.49 / 0.73 | 0.49 / 0.73 | 0.49 / 0.73 |
| TCN-acausal | 1.48 / 1.72 | 3.06 / 3.33 | 3.08 / 3.42 | 4.02 / 5.16 | 7.05 / 9.17 |
| LSTM | 0.16 / 0.26 | 0.16 / 0.26 | 0.16 / 0.26 | 0.16 / 0.26 | 0.16 / 0.26 |
| BiLSTM | 3.02 / 3.27 | 7.64 / 8.66 | 17.40 / 21.23 | 31.68 / 48.82 | 80.09 / 84.91 |

Latency measured on `x86_64`; online acausal models re-run on the last 10 s every 8 frames (250 ms), as a deployed system would have to.

Latency note: latency block from the original run on an idle machine (same model configurations); the re-run that produced these accuracy rows shared the CPU with other jobs.
