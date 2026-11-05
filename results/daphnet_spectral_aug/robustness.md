# Calibration, abstention and sensor-fault robustness (Daphnet, real data)

Seeds [0, 1, 2] · models loaded from the main benchmark (no retraining). The temperature, the decision threshold and the abstention threshold were all chosen on the validation subjects of each fold, on clean data. Faults are applied to the raw test signals only. Event metrics are pooled over test subjects within a seed (mean ± std across seeds).

## Calibration (temperature scaling fitted on validation subjects)

ECE uses 15 equal-width bins on the test subjects (mean ± std across seeds and folds). "Freeze ECE" is the calibration of p(freeze) itself.

| Model | Temperature T | ECE before | ECE after | Freeze ECE before | Freeze ECE after | NLL before | NLL after |
|---|---|---|---|---|---|---|---|
| TCN-causal | 2.25 ± 1.18 | 7.47 ± 5.34 | 7.07 ± 3.59 | 11.08 ± 6.65 | 11.73 ± 7.04 | 0.411 ± 0.216 | 0.350 ± 0.164 |

(ECE in percent.)

## Abstention: coverage vs. event-level performance (clean test data)

A frame is "uncertain" when max(p, 1-p) < tau after temperature scaling; uncertain frames raise no alarm. Coverage is the fraction of frames with a decision.

| Model | tau | Coverage (%) | Accuracy on covered frames (%) | Event F1 | Event recall | False alarms / min | Missed freezes | False alarms |
|---|---|---|---|---|---|---|---|---|
| TCN-causal | 0.5 | 100.0 ± 0.0 | 88.1 ± 0.9 | 51.7 ± 4.4 | 73.8 ± 2.9 | 0.91 ± 0.09 | 62 | 205 |
| TCN-causal | 0.6 | 92.1 ± 1.6 | 89.8 ± 0.9 | 54.4 ± 4.0 | 67.7 ± 2.2 | 0.71 ± 0.10 | 77 | 161 |
| TCN-causal | 0.7 | 83.4 ± 3.0 | 91.5 ± 1.2 | 49.8 ± 8.2 | 52.6 ± 6.7 | 0.47 ± 0.07 | 112 | 107 |
| TCN-causal | 0.8 | 72.5 ± 2.9 | 93.8 ± 1.0 | 47.0 ± 5.4 | 39.8 ± 4.8 | 0.28 ± 0.05 | 143 | 63 |
| TCN-causal | 0.9 | 59.7 ± 3.7 | 95.7 ± 0.6 | 33.3 ± 7.3 | 22.9 ± 6.0 | 0.14 ± 0.01 | 183 | 31 |

Operating abstention threshold (best validation event F1 with coverage >= 80%; the most frequent choice over folds and seeds is shown, each fold uses its own): TCN-causal 0.5.

## Sensor-fault robustness (online recognition, fixed thresholds from clean validation data)

Dropped frames hold the last received frame. Noise is Gaussian with std = factor x each feature's training-set std. A stuck sensor repeats its value for the given number of seconds (one event per 30 s of recording at random times). "Missed" = freeze episodes never detected; "FA" = false alarms (counts summed over the 8 test subjects, mean over seeds; the clean test set has 237 episodes).

### Event F1

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 52.1 ± 2.6 | 51.7 ± 4.4 |
| drop 5% | 51.5 ± 2.4 | 51.9 ± 3.5 |
| drop 10% | 51.4 ± 4.1 | 52.6 ± 4.2 |
| drop 20% | 49.9 ± 2.5 | 52.6 ± 4.7 |
| noise 0.25x std | 50.8 ± 2.1 | 48.7 ± 5.2 |
| noise 0.5x std | 36.2 ± 2.9 | 39.5 ± 7.4 |
| noise 1.0x std | 30.0 ± 1.8 | 30.2 ± 3.9 |
| stuck 1 s | 51.8 ± 1.7 | 51.5 ± 4.5 |
| stuck 2 s | 51.6 ± 3.5 | 52.0 ± 4.0 |
| stuck 3 s | 51.0 ± 2.2 | 51.8 ± 2.9 |

### Event recall (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 57.8 ± 2.8 | 73.8 ± 2.9 |
| drop 5% | 57.2 ± 3.5 | 72.3 ± 3.5 |
| drop 10% | 56.4 ± 4.8 | 72.6 ± 3.6 |
| drop 20% | 51.3 ± 2.1 | 70.3 ± 3.2 |
| noise 0.25x std | 59.1 ± 3.3 | 75.5 ± 3.1 |
| noise 0.5x std | 66.9 ± 1.0 | 78.1 ± 5.1 |
| noise 1.0x std | 72.3 ± 4.5 | 82.3 ± 4.2 |
| stuck 1 s | 57.1 ± 2.6 | 73.6 ± 3.1 |
| stuck 2 s | 56.8 ± 3.3 | 73.4 ± 3.5 |
| stuck 3 s | 56.1 ± 2.1 | 68.9 ± 3.8 |

### False alarms per minute

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 0.68 ± 0.04 | 0.91 ± 0.09 |
| drop 5% | 0.67 ± 0.04 | 0.88 ± 0.05 |
| drop 10% | 0.67 ± 0.04 | 0.89 ± 0.08 |
| drop 20% | 0.62 ± 0.03 | 0.90 ± 0.08 |
| noise 0.25x std | 0.80 ± 0.10 | 1.11 ± 0.13 |
| noise 0.5x std | 2.38 ± 0.35 | 1.73 ± 0.27 |
| noise 1.0x std | 3.47 ± 0.20 | 2.87 ± 0.23 |
| stuck 1 s | 0.68 ± 0.05 | 0.94 ± 0.09 |
| stuck 2 s | 0.66 ± 0.03 | 0.93 ± 0.10 |
| stuck 3 s | 0.69 ± 0.02 | 0.91 ± 0.09 |

### Frame accuracy (%)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 88.3 ± 0.3 | 88.1 ± 0.9 |
| drop 5% | 88.3 ± 0.3 | 88.1 ± 0.9 |
| drop 10% | 88.2 ± 0.3 | 88.1 ± 0.8 |
| drop 20% | 87.9 ± 0.3 | 88.0 ± 0.6 |
| noise 0.25x std | 88.2 ± 0.2 | 86.8 ± 1.6 |
| noise 0.5x std | 83.6 ± 1.4 | 83.9 ± 2.4 |
| noise 1.0x std | 69.0 ± 1.9 | 79.2 ± 3.3 |
| stuck 1 s | 88.2 ± 0.3 | 88.1 ± 0.9 |
| stuck 2 s | 88.3 ± 0.2 | 88.1 ± 0.9 |
| stuck 3 s | 88.3 ± 0.3 | 88.3 ± 0.6 |

### Missed freezes vs. false alarms (counts; plain decision / with abstention)

| Condition | Freeze Index | TCN-causal |
|---|---|---|
| clean | 100 / 152 | 62 / 205 ; 62 / 206 |
| drop 5% | 101 / 151 | 66 / 197 ; 66 / 199 |
| drop 10% | 103 / 150 | 65 / 201 ; 65 / 202 |
| drop 20% | 115 / 139 | 70 / 203 ; 70 / 204 |
| noise 0.25x std | 97 / 181 | 58 / 250 ; 58 / 251 |
| noise 0.5x std | 78 / 536 | 52 / 390 ; 52 / 392 |
| noise 1.0x std | 66 / 783 | 42 / 646 ; 42 / 648 |
| stuck 1 s | 102 / 152 | 63 / 211 ; 63 / 212 |
| stuck 2 s | 102 / 149 | 63 / 210 ; 63 / 211 |
| stuck 3 s | 104 / 156 | 74 / 206 ; 74 / 206 |

Each cell: missed / false alarms.
