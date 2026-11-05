# Calibration, abstention and sensor-fault robustness (Daphnet, real data)

Seeds [0, 1, 2] · models loaded from the main benchmark (no retraining). The temperature, the decision threshold and the abstention threshold were all chosen on the validation subjects of each fold, on clean data. Faults are applied to the raw test signals only. Event metrics are pooled over test subjects within a seed (mean ± std across seeds).

## Calibration (temperature scaling fitted on validation subjects)

ECE uses 15 equal-width bins on the test subjects (mean ± std across seeds and folds). "Freeze ECE" is the calibration of p(freeze) itself.

| Model | Temperature T | ECE before | ECE after | Freeze ECE before | Freeze ECE after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|
| TCN-causal | 2.13 ± 1.01 | 8.83 ± 7.81 | 7.39 ± 4.07 | 12.83 ± 9.80 | 12.73 ± 8.81 | 0.432 ± 0.309 | 0.338 ± 0.163 |

(ECE in percent.)

## Abstention: coverage vs. event-level performance (clean test data)

A frame is "uncertain" when max(p, 1-p) < tau after temperature scaling; uncertain frames raise no alarm. Coverage is the fraction of frames with a decision.

| Model | tau | Coverage (%) | Accuracy on covered frames (%) | Event F1 | Event recall | False alarms / min | Missed freezes | False alarms |
|---|---|---|---|---|---|---|---|---|
| TCN-causal | 0.5 | 100.0 ± 0.0 | 87.0 ± 1.9 | 60.4 ± 2.0 | 72.6 ± 0.9 | 0.64 ± 0.04 | 65 | 145 |
| TCN-causal | 0.6 | 92.1 ± 1.2 | 88.7 ± 1.7 | 58.4 ± 1.2 | 68.8 ± 2.6 | 0.65 ± 0.08 | 74 | 146 |
| TCN-causal | 0.7 | 83.2 ± 2.2 | 91.3 ± 1.5 | 52.0 ± 5.3 | 55.1 ± 2.8 | 0.51 ± 0.13 | 106 | 116 |
| TCN-causal | 0.8 | 73.2 ± 2.8 | 94.2 ± 1.2 | 46.5 ± 5.4 | 43.9 ± 5.0 | 0.39 ± 0.11 | 133 | 87 |
| TCN-causal | 0.9 | 60.9 ± 3.8 | 96.7 ± 0.8 | 30.7 ± 7.1 | 23.3 ± 7.3 | 0.21 ± 0.11 | 182 | 48 |

Operating abstention threshold (best validation event F1 with coverage >= 80%; the most frequent choice over folds and seeds is shown, each fold uses its own): TCN-causal 0.5.

## Sensor-fault robustness (online recognition, fixed thresholds from clean validation data)

Dropped frames hold the last received frame. Noise is Gaussian with std = factor x each feature's training-set std. A stuck sensor repeats its value for the given number of seconds (one event per 30 s of recording at random times). "Missed" = freeze episodes never detected; "FA" = false alarms (counts summed over the 8 test subjects, mean over seeds; the clean test set has 237 episodes).

### Event F1

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 52.1 ± 2.6 | 60.4 ± 2.0 |
| drop 5% | 51.5 ± 2.4 | 58.9 ± 0.7 |
| drop 10% | 51.4 ± 4.1 | 59.1 ± 1.3 |
| drop 20% | 49.9 ± 2.5 | 58.5 ± 2.7 |
| noise 0.25x std | 50.8 ± 2.1 | 51.2 ± 1.5 |
| noise 0.5x std | 36.2 ± 2.9 | 41.8 ± 3.0 |
| noise 1.0x std | 30.0 ± 1.8 | 33.7 ± 2.2 |
| stuck 1 s | 51.8 ± 1.7 | 58.5 ± 2.0 |
| stuck 2 s | 51.6 ± 3.5 | 57.9 ± 1.1 |
| stuck 3 s | 51.0 ± 2.2 | 56.2 ± 1.2 |

### Event recall (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 57.8 ± 2.8 | 72.6 ± 0.9 |
| drop 5% | 57.2 ± 3.5 | 70.5 ± 0.6 |
| drop 10% | 56.4 ± 4.8 | 68.6 ± 1.0 |
| drop 20% | 51.3 ± 2.1 | 66.4 ± 1.3 |
| noise 0.25x std | 59.1 ± 3.3 | 78.8 ± 2.9 |
| noise 0.5x std | 66.9 ± 1.0 | 83.5 ± 2.3 |
| noise 1.0x std | 72.3 ± 4.5 | 88.5 ± 3.8 |
| stuck 1 s | 57.1 ± 2.6 | 71.2 ± 0.2 |
| stuck 2 s | 56.8 ± 3.3 | 69.8 ± 0.5 |
| stuck 3 s | 56.1 ± 2.1 | 63.6 ± 0.9 |

### False alarms per minute

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 0.68 ± 0.04 | 0.64 ± 0.04 |
| drop 5% | 0.67 ± 0.04 | 0.67 ± 0.01 |
| drop 10% | 0.67 ± 0.04 | 0.63 ± 0.01 |
| drop 20% | 0.62 ± 0.03 | 0.66 ± 0.07 |
| noise 0.25x std | 0.80 ± 0.10 | 1.15 ± 0.06 |
| noise 0.5x std | 2.38 ± 0.35 | 2.00 ± 0.27 |
| noise 1.0x std | 3.47 ± 0.20 | 3.25 ± 0.41 |
| stuck 1 s | 0.68 ± 0.05 | 0.70 ± 0.06 |
| stuck 2 s | 0.66 ± 0.03 | 0.71 ± 0.04 |
| stuck 3 s | 0.69 ± 0.02 | 0.72 ± 0.08 |

### Frame accuracy (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 88.3 ± 0.3 | 87.0 ± 1.9 |
| drop 5% | 88.3 ± 0.3 | 87.1 ± 1.8 |
| drop 10% | 88.2 ± 0.3 | 87.1 ± 1.8 |
| drop 20% | 87.9 ± 0.3 | 87.0 ± 1.6 |
| noise 0.25x std | 88.2 ± 0.2 | 83.2 ± 3.6 |
| noise 0.5x std | 83.6 ± 1.4 | 80.1 ± 3.6 |
| noise 1.0x std | 69.0 ± 1.9 | 78.9 ± 3.0 |
| stuck 1 s | 88.2 ± 0.3 | 87.1 ± 1.7 |
| stuck 2 s | 88.3 ± 0.2 | 87.1 ± 1.7 |
| stuck 3 s | 88.3 ± 0.3 | 87.5 ± 1.2 |

### Missed freezes vs. false alarms (counts; plain decision / with abstention)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 100 / 152 | 65 / 145 ; 65 / 145 |
| drop 5% | 101 / 151 | 70 / 150 ; 70 / 150 |
| drop 10% | 103 / 150 | 74 / 143 ; 75 / 143 |
| drop 20% | 115 / 139 | 80 / 148 ; 80 / 148 |
| noise 0.25x std | 97 / 181 | 50 / 259 ; 50 / 260 |
| noise 0.5x std | 78 / 536 | 39 / 451 ; 40 / 453 |
| noise 1.0x std | 66 / 783 | 27 / 732 ; 29 / 730 |
| stuck 1 s | 102 / 152 | 68 / 157 ; 69 / 156 |
| stuck 2 s | 102 / 149 | 72 / 159 ; 72 / 158 |
| stuck 3 s | 104 / 156 | 86 / 162 ; 86 / 162 |

Each cell: missed / false alarms.
