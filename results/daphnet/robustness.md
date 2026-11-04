# Calibration, abstention and sensor-fault robustness (Daphnet, real data)

Seeds [0, 1, 2] · models loaded from the main benchmark (no retraining). The temperature, the decision threshold and the abstention threshold were all chosen on the validation subjects of each fold, on clean data. Faults are applied to the raw test signals only. Event metrics are pooled over test subjects within a seed (mean ± std across seeds).

## Calibration (temperature scaling fitted on validation subjects)

ECE uses 15 equal-width bins on the test subjects (mean ± std across seeds and folds). "Freeze ECE" is the calibration of p(freeze) itself.

| Model | Temperature T | ECE before | ECE after | Freeze ECE before | Freeze ECE after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|
| TCN-causal | 2.63 ± 1.37 | 7.42 ± 4.62 | 8.38 ± 4.41 | 10.63 ± 5.75 | 12.23 ± 7.77 | 0.421 ± 0.240 | 0.372 ± 0.170 |
| TCN-acausal | 1.93 ± 1.24 | 6.91 ± 5.15 | 8.38 ± 5.38 | 10.79 ± 7.54 | 12.90 ± 8.60 | 0.378 ± 0.190 | 0.379 ± 0.164 |
| LSTM | 2.03 ± 0.60 | 12.88 ± 7.87 | 12.49 ± 6.21 | 16.00 ± 8.71 | 17.17 ± 8.34 | 0.526 ± 0.251 | 0.446 ± 0.135 |
| BiLSTM | 1.56 ± 0.54 | 12.95 ± 8.11 | 12.95 ± 7.63 | 15.88 ± 9.14 | 16.62 ± 9.33 | 0.455 ± 0.164 | 0.437 ± 0.131 |

(ECE in percent.)

## Abstention: coverage vs. event-level performance (clean test data)

A frame is "uncertain" when max(p, 1-p) < tau after temperature scaling; uncertain frames raise no alarm. Coverage is the fraction of frames with a decision.

| Model | tau | Coverage (%) | Accuracy on covered frames (%) | Event F1 | Event recall | False alarms / min | Missed freezes | False alarms |
|---|---|---|---|---|---|---|---|---|
| TCN-causal | 0.5 | 100.0 ± 0.0 | 83.4 ± 5.5 | 48.8 ± 4.2 | 65.7 ± 9.0 | 1.14 ± 0.09 | 81 | 256 |
| TCN-causal | 0.6 | 88.5 ± 4.5 | 87.7 ± 2.5 | 44.9 ± 4.3 | 54.6 ± 9.2 | 1.02 ± 0.49 | 108 | 230 |
| TCN-causal | 0.7 | 77.9 ± 8.5 | 92.5 ± 2.2 | 44.5 ± 3.2 | 39.8 ± 0.9 | 0.49 ± 0.14 | 143 | 111 |
| TCN-causal | 0.8 | 68.0 ± 9.7 | 94.0 ± 2.0 | 33.8 ± 7.1 | 24.8 ± 7.2 | 0.22 ± 0.09 | 178 | 49 |
| TCN-causal | 0.9 | 54.7 ± 9.9 | 95.4 ± 1.6 | 18.2 ± 10.6 | 11.7 ± 7.6 | 0.11 ± 0.08 | 209 | 24 |
| TCN-acausal | 0.5 | 100.0 ± 0.0 | 81.1 ± 6.9 | 47.8 ± 6.6 | 64.1 ± 12.5 | 0.94 ± 0.22 | 85 | 212 |
| TCN-acausal | 0.6 | 87.7 ± 3.3 | 85.9 ± 4.8 | 44.0 ± 8.1 | 55.3 ± 14.7 | 1.32 ± 0.94 | 106 | 297 |
| TCN-acausal | 0.7 | 76.9 ± 6.3 | 90.8 ± 1.2 | 39.6 ± 4.4 | 38.5 ± 3.9 | 0.76 ± 0.49 | 146 | 170 |
| TCN-acausal | 0.8 | 65.6 ± 8.5 | 93.8 ± 1.4 | 31.2 ± 10.8 | 22.4 ± 8.7 | 0.24 ± 0.02 | 184 | 54 |
| TCN-acausal | 0.9 | 52.3 ± 8.4 | 95.5 ± 0.9 | 18.7 ± 12.5 | 11.7 ± 8.1 | 0.07 ± 0.04 | 209 | 16 |
| LSTM | 0.5 | 100.0 ± 0.0 | 72.2 ± 1.6 | 25.5 ± 2.6 | 58.1 ± 5.5 | 1.53 ± 0.32 | 99 | 344 |
| LSTM | 0.6 | 90.8 ± 0.6 | 76.5 ± 2.6 | 27.9 ± 3.0 | 51.1 ± 9.0 | 1.65 ± 0.85 | 116 | 372 |
| LSTM | 0.7 | 76.5 ± 0.6 | 81.3 ± 2.4 | 28.5 ± 1.4 | 41.9 ± 6.8 | 1.74 ± 0.92 | 138 | 393 |
| LSTM | 0.8 | 60.1 ± 3.4 | 88.1 ± 3.7 | 14.5 ± 12.1 | 12.2 ± 10.5 | 0.43 ± 0.34 | 208 | 97 |
| LSTM | 0.9 | 38.8 ± 1.1 | 94.1 ± 1.4 | 5.1 ± 7.2 | 2.8 ± 4.0 | 0.01 ± 0.02 | 230 | 3 |
| BiLSTM | 0.5 | 100.0 ± 0.0 | 77.5 ± 1.6 | 36.3 ± 2.5 | 60.9 ± 8.7 | 1.58 ± 0.45 | 93 | 357 |
| BiLSTM | 0.6 | 91.6 ± 2.3 | 79.4 ± 2.0 | 38.2 ± 3.2 | 58.1 ± 6.0 | 1.55 ± 0.67 | 99 | 350 |
| BiLSTM | 0.7 | 77.3 ± 5.3 | 83.6 ± 2.5 | 34.9 ± 7.2 | 41.5 ± 1.2 | 1.40 ± 0.60 | 139 | 316 |
| BiLSTM | 0.8 | 56.4 ± 4.3 | 93.2 ± 3.7 | 22.3 ± 7.7 | 15.2 ± 6.4 | 0.33 ± 0.17 | 201 | 74 |
| BiLSTM | 0.9 | 32.8 ± 10.9 | 97.1 ± 3.1 | 2.2 ± 1.0 | 1.1 ± 0.5 | 0.00 ± 0.00 | 234 | 1 |

Operating abstention threshold (best validation event F1 with coverage >= 80%; the most frequent choice over folds and seeds is shown, each fold uses its own): TCN-causal 0.5, TCN-acausal 0.5, LSTM 0.5, BiLSTM 0.5.

## Sensor-fault robustness (online recognition, fixed thresholds from clean validation data)

Dropped frames hold the last received frame. Noise is Gaussian with std = factor x each feature's training-set std. A stuck sensor repeats its value for the given number of seconds (one event per 30 s of recording at random times). "Missed" = freeze episodes never detected; "FA" = false alarms (counts summed over the 8 test subjects, mean over seeds; the clean test set has 237 episodes).

### Event F1

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 52.1 ± 2.6 | 48.8 ± 4.2 | 47.8 ± 6.6 | 25.5 ± 2.6 | 36.3 ± 2.5 |
| drop 5% | 51.5 ± 2.4 | 49.1 ± 3.8 | 46.8 ± 6.8 | 24.9 ± 1.5 | 36.2 ± 1.1 |
| drop 10% | 51.4 ± 4.1 | 49.2 ± 3.7 | 45.3 ± 5.3 | 25.3 ± 2.7 | 35.8 ± 3.0 |
| drop 20% | 49.9 ± 2.5 | 48.7 ± 2.9 | 45.0 ± 5.4 | 27.5 ± 2.1 | 36.7 ± 0.8 |
| noise 0.25x std | 50.8 ± 2.1 | 46.3 ± 2.7 | 45.3 ± 4.4 | 25.8 ± 2.7 | 34.3 ± 1.4 |
| noise 0.5x std | 36.2 ± 2.9 | 42.2 ± 1.9 | 40.5 ± 3.7 | 26.2 ± 1.2 | 32.5 ± 1.7 |
| noise 1.0x std | 30.0 ± 1.8 | 35.5 ± 3.3 | 34.9 ± 0.6 | 27.0 ± 1.6 | 31.1 ± 3.7 |
| stuck 1 s | 51.8 ± 1.7 | 48.4 ± 3.9 | 46.7 ± 6.6 | 25.9 ± 3.7 | 35.6 ± 2.1 |
| stuck 2 s | 51.6 ± 3.5 | 47.7 ± 3.7 | 45.6 ± 7.3 | 25.0 ± 1.6 | 35.0 ± 2.5 |
| stuck 3 s | 51.0 ± 2.2 | 47.0 ± 3.8 | 46.2 ± 7.3 | 25.1 ± 2.6 | 35.9 ± 2.4 |

### Event recall (%)

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 57.8 ± 2.8 | 65.7 ± 9.0 | 64.1 ± 12.5 | 58.1 ± 5.5 | 60.9 ± 8.7 |
| drop 5% | 57.2 ± 3.5 | 66.8 ± 8.5 | 63.4 ± 12.8 | 59.4 ± 6.7 | 61.0 ± 8.0 |
| drop 10% | 56.4 ± 4.8 | 66.0 ± 8.8 | 61.0 ± 13.5 | 58.6 ± 5.9 | 60.6 ± 8.1 |
| drop 20% | 51.3 ± 2.1 | 63.6 ± 8.4 | 59.9 ± 14.2 | 61.7 ± 5.9 | 62.3 ± 9.1 |
| noise 0.25x std | 59.1 ± 3.3 | 72.9 ± 7.4 | 68.9 ± 11.4 | 60.2 ± 6.2 | 63.0 ± 10.7 |
| noise 0.5x std | 66.9 ± 1.0 | 81.7 ± 5.0 | 76.1 ± 6.8 | 62.4 ± 7.4 | 67.7 ± 14.1 |
| noise 1.0x std | 72.3 ± 4.5 | 91.8 ± 3.1 | 81.0 ± 6.3 | 72.7 ± 16.4 | 78.5 ± 10.0 |
| stuck 1 s | 57.1 ± 2.6 | 65.0 ± 8.5 | 63.9 ± 13.0 | 59.6 ± 6.2 | 60.8 ± 8.0 |
| stuck 2 s | 56.8 ± 3.3 | 64.3 ± 8.9 | 63.3 ± 12.1 | 58.1 ± 5.2 | 61.3 ± 7.8 |
| stuck 3 s | 56.1 ± 2.1 | 63.0 ± 8.9 | 62.9 ± 12.5 | 59.1 ± 5.5 | 61.9 ± 7.8 |

### False alarms per minute

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 0.68 ± 0.04 | 1.14 ± 0.09 | 0.94 ± 0.22 | 1.53 ± 0.32 | 1.58 ± 0.45 |
| drop 5% | 0.67 ± 0.04 | 1.16 ± 0.07 | 0.96 ± 0.24 | 1.58 ± 0.31 | 1.61 ± 0.47 |
| drop 10% | 0.67 ± 0.04 | 1.12 ± 0.11 | 0.99 ± 0.21 | 1.69 ± 0.38 | 1.63 ± 0.48 |
| drop 20% | 0.62 ± 0.03 | 1.09 ± 0.11 | 1.02 ± 0.22 | 1.75 ± 0.36 | 1.68 ± 0.47 |
| noise 0.25x std | 0.80 ± 0.10 | 1.50 ± 0.16 | 1.28 ± 0.13 | 1.81 ± 0.41 | 2.02 ± 0.63 |
| noise 0.5x std | 2.38 ± 0.35 | 2.18 ± 0.24 | 2.01 ± 0.13 | 2.46 ± 0.68 | 2.60 ± 0.82 |
| noise 1.0x std | 3.47 ± 0.20 | 3.36 ± 0.41 | 3.14 ± 0.28 | 3.11 ± 0.77 | 3.19 ± 1.09 |
| stuck 1 s | 0.68 ± 0.05 | 1.15 ± 0.07 | 1.01 ± 0.22 | 1.61 ± 0.35 | 1.68 ± 0.48 |
| stuck 2 s | 0.66 ± 0.03 | 1.16 ± 0.09 | 1.04 ± 0.27 | 1.62 ± 0.30 | 1.68 ± 0.44 |
| stuck 3 s | 0.69 ± 0.02 | 1.14 ± 0.08 | 1.01 ± 0.28 | 1.57 ± 0.31 | 1.63 ± 0.44 |

### Frame accuracy (%)

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 88.3 ± 0.3 | 83.4 ± 5.5 | 81.1 ± 6.9 | 72.2 ± 1.6 | 77.5 ± 1.6 |
| drop 5% | 88.3 ± 0.3 | 83.4 ± 5.5 | 81.1 ± 6.9 | 72.2 ± 1.6 | 77.5 ± 1.5 |
| drop 10% | 88.2 ± 0.3 | 83.4 ± 5.5 | 81.1 ± 6.9 | 72.2 ± 1.7 | 77.4 ± 1.6 |
| drop 20% | 87.9 ± 0.3 | 83.2 ± 5.6 | 81.2 ± 6.9 | 72.2 ± 1.9 | 77.3 ± 1.5 |
| noise 0.25x std | 88.2 ± 0.2 | 82.3 ± 6.0 | 79.7 ± 8.0 | 73.7 ± 2.2 | 77.3 ± 1.7 |
| noise 0.5x std | 83.6 ± 1.4 | 79.7 ± 6.7 | 76.7 ± 8.3 | 75.6 ± 3.4 | 76.9 ± 2.3 |
| noise 1.0x std | 69.0 ± 1.9 | 76.6 ± 6.7 | 73.5 ± 7.9 | 74.9 ± 4.0 | 73.0 ± 3.5 |
| stuck 1 s | 88.2 ± 0.3 | 83.3 ± 5.5 | 81.0 ± 7.0 | 72.0 ± 1.7 | 77.3 ± 1.5 |
| stuck 2 s | 88.3 ± 0.2 | 83.3 ± 5.3 | 80.9 ± 6.8 | 71.8 ± 1.7 | 77.0 ± 1.6 |
| stuck 3 s | 88.3 ± 0.3 | 83.4 ± 5.3 | 81.0 ± 6.8 | 71.1 ± 1.9 | 76.9 ± 1.5 |

### Missed freezes vs. false alarms (counts; plain decision / with abstention)

| Condition | Freeze Index | TCN-causal | TCN-acausal | LSTM | BiLSTM |
|---|---|---|---|---|---|
| clean | 100 / 152 | 81 / 256 ; 81 / 256 | 85 / 212 ; 85 / 212 | 99 / 344 ; 104 / 391 | 93 / 357 ; 94 / 407 |
| drop 5% | 101 / 151 | 79 / 261 ; 79 / 261 | 87 / 217 ; 87 / 217 | 96 / 355 ; 101 / 404 | 92 / 364 ; 94 / 412 |
| drop 10% | 103 / 150 | 81 / 253 ; 81 / 253 | 92 / 224 ; 92 / 224 | 98 / 381 ; 101 / 430 | 93 / 368 ; 94 / 413 |
| drop 20% | 115 / 139 | 86 / 245 ; 86 / 245 | 95 / 230 ; 95 / 230 | 91 / 395 ; 94 / 445 | 89 / 379 ; 91 / 422 |
| noise 0.25x std | 97 / 181 | 64 / 338 ; 64 / 338 | 74 / 288 ; 74 / 290 | 94 / 408 ; 97 / 452 | 88 / 456 ; 89 / 504 |
| noise 0.5x std | 78 / 536 | 43 / 491 ; 43 / 491 | 57 / 453 ; 57 / 467 | 89 / 554 ; 90 / 587 | 77 / 585 ; 78 / 640 |
| noise 1.0x std | 66 / 783 | 19 / 758 ; 19 / 758 | 45 / 708 ; 46 / 722 | 65 / 701 ; 65 / 729 | 51 / 720 ; 52 / 776 |
| stuck 1 s | 102 / 152 | 83 / 258 ; 83 / 258 | 86 / 229 ; 86 / 229 | 96 / 364 ; 101 / 414 | 93 / 378 ; 95 / 426 |
| stuck 2 s | 102 / 149 | 85 / 262 ; 85 / 262 | 87 / 234 ; 87 / 235 | 99 / 365 ; 104 / 416 | 92 / 378 ; 93 / 426 |
| stuck 3 s | 104 / 156 | 88 / 256 ; 88 / 256 | 88 / 227 ; 88 / 228 | 97 / 355 ; 102 / 405 | 90 / 368 ; 92 / 414 |

Each cell: missed / false alarms.
