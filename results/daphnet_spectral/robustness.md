# Calibration, abstention and sensor-fault robustness (Daphnet, real data)

Seeds [0, 1, 2] · models loaded from the main benchmark (no retraining). The temperature, the decision threshold and the abstention threshold were all chosen on the validation subjects of each fold, on clean data. Faults are applied to the raw test signals only. Event metrics are pooled over test subjects within a seed (mean ± std across seeds).

## Calibration (temperature scaling fitted on validation subjects)

ECE uses 15 equal-width bins on the test subjects (mean ± std across seeds and folds). "Freeze ECE" is the calibration of p(freeze) itself.

| Model | Temperature T | ECE before | ECE after | Freeze ECE before | Freeze ECE after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|
| TCN-causal | 2.00 ± 0.93 | 8.42 ± 8.30 | 7.91 ± 5.38 | 12.62 ± 11.31 | 13.43 ± 10.11 | 0.382 ± 0.222 | 0.341 ± 0.170 |

(ECE in percent.)

## Abstention: coverage vs. event-level performance (clean test data)

A frame is "uncertain" when max(p, 1-p) < tau after temperature scaling; uncertain frames raise no alarm. Coverage is the fraction of frames with a decision.

| Model | tau | Coverage (%) | Accuracy on covered frames (%) | Event F1 | Event recall | False alarms / min | Missed freezes | False alarms |
|---|---|---|---|---|---|---|---|---|
| TCN-causal | 0.5 | 100.0 ± 0.0 | 87.9 ± 1.4 | 56.8 ± 2.6 | 66.7 ± 3.0 | 0.62 ± 0.13 | 79 | 139 |
| TCN-causal | 0.6 | 91.0 ± 0.8 | 89.5 ± 1.4 | 55.2 ± 4.5 | 61.7 ± 1.6 | 0.58 ± 0.16 | 91 | 130 |
| TCN-causal | 0.7 | 80.9 ± 1.3 | 92.3 ± 1.0 | 54.7 ± 5.4 | 55.7 ± 1.8 | 0.48 ± 0.16 | 105 | 108 |
| TCN-causal | 0.8 | 71.1 ± 2.0 | 94.5 ± 1.1 | 50.4 ± 4.8 | 44.3 ± 1.2 | 0.31 ± 0.16 | 132 | 71 |
| TCN-causal | 0.9 | 59.9 ± 3.1 | 96.6 ± 0.6 | 36.7 ± 2.2 | 26.3 ± 3.5 | 0.16 ± 0.06 | 175 | 35 |

Operating abstention threshold (best validation event F1 with coverage >= 80%; the most frequent choice over folds and seeds is shown, each fold uses its own): TCN-causal 0.5.

## Sensor-fault robustness (online recognition, fixed thresholds from clean validation data)

Dropped frames hold the last received frame. Noise is Gaussian with std = factor x each feature's training-set std. A stuck sensor repeats its value for the given number of seconds (one event per 30 s of recording at random times). "Missed" = freeze episodes never detected; "FA" = false alarms (counts summed over the 8 test subjects, mean over seeds; the clean test set has 237 episodes).

### Event F1

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 52.1 ± 2.6 | 56.8 ± 2.6 |
| drop 5% | 51.5 ± 2.4 | 56.4 ± 2.0 |
| drop 10% | 51.4 ± 4.1 | 56.0 ± 2.7 |
| drop 20% | 49.9 ± 2.5 | 54.5 ± 1.5 |
| noise 0.25x std | 50.8 ± 2.1 | 46.8 ± 2.4 |
| noise 0.5x std | 36.2 ± 2.9 | 39.5 ± 2.4 |
| noise 1.0x std | 30.0 ± 1.8 | 33.0 ± 3.3 |
| stuck 1 s | 51.8 ± 1.7 | 56.1 ± 2.4 |
| stuck 2 s | 51.6 ± 3.5 | 56.0 ± 2.3 |
| stuck 3 s | 51.0 ± 2.2 | 54.1 ± 2.2 |

### Event recall (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 57.8 ± 2.8 | 66.7 ± 3.0 |
| drop 5% | 57.2 ± 3.5 | 65.4 ± 2.5 |
| drop 10% | 56.4 ± 4.8 | 63.4 ± 3.5 |
| drop 20% | 51.3 ± 2.1 | 60.1 ± 2.8 |
| noise 0.25x std | 59.1 ± 3.3 | 72.0 ± 4.3 |
| noise 0.5x std | 66.9 ± 1.0 | 75.0 ± 3.6 |
| noise 1.0x std | 72.3 ± 4.5 | 84.1 ± 3.4 |
| stuck 1 s | 57.1 ± 2.6 | 65.3 ± 2.3 |
| stuck 2 s | 56.8 ± 3.3 | 65.1 ± 2.9 |
| stuck 3 s | 56.1 ± 2.1 | 60.5 ± 3.7 |

### False alarms per minute

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 0.68 ± 0.04 | 0.62 ± 0.13 |
| drop 5% | 0.67 ± 0.04 | 0.61 ± 0.12 |
| drop 10% | 0.67 ± 0.04 | 0.60 ± 0.13 |
| drop 20% | 0.62 ± 0.03 | 0.60 ± 0.13 |
| noise 0.25x std | 0.80 ± 0.10 | 1.08 ± 0.17 |
| noise 0.5x std | 2.38 ± 0.35 | 1.64 ± 0.13 |
| noise 1.0x std | 3.47 ± 0.20 | 3.30 ± 0.49 |
| stuck 1 s | 0.68 ± 0.05 | 0.62 ± 0.12 |
| stuck 2 s | 0.66 ± 0.03 | 0.64 ± 0.12 |
| stuck 3 s | 0.69 ± 0.02 | 0.66 ± 0.18 |

### Frame accuracy (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 88.3 ± 0.3 | 87.9 ± 1.4 |
| drop 5% | 88.3 ± 0.3 | 88.0 ± 1.4 |
| drop 10% | 88.2 ± 0.3 | 88.0 ± 1.4 |
| drop 20% | 87.9 ± 0.3 | 88.0 ± 1.3 |
| noise 0.25x std | 88.2 ± 0.2 | 82.4 ± 1.9 |
| noise 0.5x std | 83.6 ± 1.4 | 80.0 ± 2.6 |
| noise 1.0x std | 69.0 ± 1.9 | 79.1 ± 2.7 |
| stuck 1 s | 88.2 ± 0.3 | 87.9 ± 1.5 |
| stuck 2 s | 88.3 ± 0.2 | 87.9 ± 1.4 |
| stuck 3 s | 88.3 ± 0.3 | 88.3 ± 1.0 |

### Missed freezes vs. false alarms (counts; plain decision / with abstention)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 100 / 152 | 79 / 139 ; 79 / 139 |
| drop 5% | 101 / 151 | 82 / 138 ; 82 / 138 |
| drop 10% | 103 / 150 | 87 / 135 ; 87 / 135 |
| drop 20% | 115 / 139 | 95 / 136 ; 95 / 136 |
| noise 0.25x std | 97 / 181 | 66 / 243 ; 66 / 243 |
| noise 0.5x std | 78 / 536 | 59 / 370 ; 59 / 370 |
| noise 1.0x std | 66 / 783 | 38 / 744 ; 38 / 744 |
| stuck 1 s | 102 / 152 | 82 / 140 ; 82 / 140 |
| stuck 2 s | 102 / 149 | 83 / 144 ; 83 / 144 |
| stuck 3 s | 104 / 156 | 94 / 148 ; 94 / 148 |

Each cell: missed / false alarms.
