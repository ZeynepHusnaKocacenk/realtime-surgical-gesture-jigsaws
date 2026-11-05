# Daphnet: headline results with uncertainty

Online (deployable) setting, 8 test subjects x 3 seeds. Point estimate = pooled over test subjects within a seed, mean over seeds. **[95% CI]** = percentile bootstrap over test subjects (2000 resamples of the 8 patients; the patient is the unit of independence). PR-AUC = mean over subjects of the per-subject value (mean over seeds). *Strict* F1 uses one-to-one event matching (`event_counts(strict=True)`), a sensitivity check of the event definition. Every threshold and debounce was chosen on validation subjects only.

**Model selection.** Among the causal TCN variants, the one with the highest mean validation PR-AUC (validation patients only) is selected: **TCN-spectral, tuned** (mean validation PR-AUC: TCN-causal 0.462, TCN-causal, deeper + noise aug. 0.486, TCN-spectral + noise aug. 0.589, TCN-spectral 0.615, TCN-spectral, tuned 0.629). Test results played no part in this choice; the other variants are shown as ablations.

| Detector | Operating point | Event F1 [95% CI] | Strict event F1 | Recall | Precision | False alarms / min | PR-AUC [95% CI] | Debounce (median) |
|---|---|---|---|---|---|---|---|---|
| Freeze Index | threshold | 52.1 [40.7, 58.2] | 40.3 [30.5, 46.4] | 57.8 | 47.5 | 0.68 | 0.414 [0.290, 0.528] | 0.00 s |
| Freeze Index | threshold+debounce | 55.0 [44.8, 60.2] | 42.0 [33.3, 48.5] | 59.6 | 51.0 | 0.59 | 0.427 [0.309, 0.535] | 0.38 s |
| TCN-causal | threshold | 48.1 [34.0, 59.3] | 36.2 [23.5, 48.6] | 65.5 | 38.3 | 1.11 | 0.528 [0.429, 0.631] | 0.00 s |
| TCN-causal | threshold+debounce | 45.1 [32.7, 56.6] | 33.7 [22.9, 45.2] | 64.8 | 35.4 | 1.11 | 0.528 [0.429, 0.631] | 0.00 s |
| TCN-acausal | threshold | 47.5 [34.1, 57.0] | 35.9 [24.6, 46.5] | 61.0 | 39.9 | 1.00 | 0.520 [0.409, 0.638] | 0.00 s |
| TCN-acausal | threshold+debounce | 47.0 [33.9, 56.8] | 35.2 [23.9, 46.2] | 58.6 | 39.9 | 0.97 | 0.520 [0.409, 0.638] | 0.25 s |
| LSTM | threshold | 26.7 [16.7, 39.5] | 18.5 [11.6, 28.4] | 57.0 | 17.6 | 1.63 | 0.376 [0.273, 0.479] | 0.00 s |
| LSTM | threshold+debounce | 25.4 [15.2, 39.2] | 16.4 [10.4, 25.7] | 58.4 | 16.4 | 1.75 | 0.376 [0.273, 0.479] | 0.25 s |
| BiLSTM | threshold | 35.7 [21.3, 49.0] | 24.6 [14.0, 36.8] | 61.6 | 25.2 | 1.60 | 0.363 [0.253, 0.486] | 0.00 s |
| BiLSTM | threshold+debounce | 31.3 [19.0, 44.4] | 20.9 [13.2, 30.0] | 56.7 | 22.5 | 1.71 | 0.363 [0.253, 0.486] | 1.00 s |
| TCN-causal, deeper + noise aug. | threshold | 47.8 [32.4, 61.0] | 35.1 [23.2, 46.6] | 63.0 | 38.7 | 0.85 | 0.496 [0.372, 0.617] | 0.00 s |
| TCN-causal, deeper + noise aug. | threshold+debounce | 45.9 [32.7, 58.6] | 34.1 [24.0, 46.0] | 57.5 | 38.3 | 0.76 | 0.496 [0.372, 0.617] | 0.00 s |
| TCN-spectral + noise aug. | threshold | 52.0 [35.9, 64.3] | 36.4 [22.7, 50.3] | 71.0 | 41.0 | 0.83 | 0.598 [0.501, 0.702] | 0.00 s |
| TCN-spectral + noise aug. | threshold+debounce | 52.8 [38.2, 63.9] | 37.7 [25.0, 49.9] | 69.8 | 42.6 | 0.77 | 0.598 [0.501, 0.702] | 0.25 s |
| TCN-spectral | threshold | 58.5 [43.7, 68.1] | 40.8 [28.0, 53.3] | 70.7 | 50.1 | 0.63 | 0.641 [0.549, 0.735] | 0.00 s |
| TCN-spectral | threshold+debounce | 53.8 [40.5, 63.9] | 36.6 [25.8, 47.4] | 66.9 | 45.0 | 0.69 | 0.641 [0.549, 0.735] | 0.00 s |
| TCN-spectral, tuned | threshold | 61.2 [48.5, 68.4] | 42.0 [30.5, 51.5] | 72.3 | 53.1 | 0.61 | 0.599 [0.489, 0.705] | 0.00 s |
| TCN-spectral, tuned | threshold+debounce | 55.4 [41.7, 64.9] | 39.3 [28.8, 49.3] | 67.4 | 47.4 | 0.72 | 0.599 [0.489, 0.705] | 0.25 s |

## Paired comparison with the Freeze Index (same operating-point rule)

Difference in event F1 (points). CI = paired bootstrap over test subjects; P(>0) = fraction of bootstrap resamples in which the learned model is better; Wilcoxon = exact two-sided signed-rank test on per-subject F1 (smallest attainable p with 8 subjects is 0.008); wins = subjects where the learned model has the higher F1. Delta PR-AUC = mean over subjects of the per-subject difference (the Freeze Index scored on its power-gated FI, the signal its detector uses).

| Detector | Operating point | Delta F1 [95% CI] | P(>0) | Wilcoxon p | Wins | Delta strict F1 [95% CI] | Delta PR-AUC [95% CI] | PR-AUC Wilcoxon p | PR-AUC wins |
|---|---|---|---|---|---|---|---|---|---|
| TCN-causal | threshold | -4.0 [-10.2, +3.1] | 0.18 | 0.945 | 4/8 | -4.1 [-11.4, +5.1] | +0.114 [+0.029, +0.193] | 0.055 | 6/8 |
| TCN-causal | threshold+debounce | -9.9 [-16.2, -0.9] | 0.02 | 0.250 | 3/8 | -8.3 [-15.2, +1.2] | +0.100 [+0.025, +0.166] | 0.055 | 6/8 |
| TCN-acausal | threshold | -4.6 [-9.3, +1.7] | 0.07 | 0.461 | 3/8 | -4.4 [-11.0, +4.4] | +0.106 [+0.029, +0.184] | 0.078 | 6/8 |
| TCN-acausal | threshold+debounce | -7.9 [-14.4, -0.2] | 0.02 | 0.148 | 3/8 | -6.8 [-14.1, +2.7] | +0.093 [+0.026, +0.161] | 0.078 | 6/8 |
| LSTM | threshold | -25.4 [-30.6, -14.5] | 0.00 | 0.016 | 1/8 | -21.8 [-25.4, -14.2] | -0.038 [-0.139, +0.060] | 0.641 | 3/8 |
| LSTM | threshold+debounce | -29.5 [-34.4, -18.6] | 0.00 | 0.016 | 1/8 | -25.6 [-29.5, -18.9] | -0.051 [-0.150, +0.047] | 0.547 | 3/8 |
| BiLSTM | threshold | -16.4 [-24.3, -5.7] | 0.00 | 0.078 | 2/8 | -15.7 [-22.5, -6.0] | -0.051 [-0.127, +0.019] | 0.641 | 4/8 |
| BiLSTM | threshold+debounce | -23.7 [-33.0, -11.2] | 0.00 | 0.039 | 2/8 | -21.1 [-28.8, -12.2] | -0.065 [-0.137, +0.003] | 0.312 | 4/8 |
| TCN-causal, deeper + noise aug. | threshold | -4.3 [-14.1, +8.0] | 0.23 | 0.383 | 4/8 | -5.2 [-14.2, +7.0] | +0.082 [+0.012, +0.158] | 0.109 | 5/8 |
| TCN-causal, deeper + noise aug. | threshold+debounce | -9.1 [-19.5, +3.2] | 0.07 | 0.312 | 3/8 | -7.9 [-17.6, +4.2] | +0.069 [+0.009, +0.127] | 0.109 | 5/8 |
| TCN-spectral + noise aug. | threshold | -0.2 [-10.1, +9.3] | 0.53 | 0.742 | 4/8 | -3.8 [-11.3, +5.9] | +0.184 [+0.110, +0.262] | 0.008 | 8/8 |
| TCN-spectral + noise aug. | threshold+debounce | -2.2 [-10.4, +5.7] | 0.33 | 0.547 | 6/8 | -4.3 [-11.1, +4.6] | +0.170 [+0.106, +0.241] | 0.008 | 8/8 |
| TCN-spectral | threshold | +6.3 [-3.0, +14.7] | 0.92 | 0.148 | 7/8 | +0.6 [-6.6, +9.2] | +0.227 [+0.141, +0.314] | 0.008 | 8/8 |
| TCN-spectral | threshold+debounce | -1.1 [-8.1, +6.4] | 0.37 | 0.945 | 5/8 | -5.4 [-10.8, +2.4] | +0.214 [+0.137, +0.293] | 0.008 | 8/8 |
| TCN-spectral, tuned | threshold | +9.1 [+2.8, +14.5] | 1.00 | 0.148 | 7/8 | +1.7 [-3.6, +7.1] | +0.185 [+0.108, +0.270] | 0.008 | 8/8 |
| TCN-spectral, tuned | threshold+debounce | +0.4 [-6.9, +7.6] | 0.57 | 0.461 | 5/8 | -2.7 [-8.2, +4.8] | +0.171 [+0.107, +0.242] | 0.008 | 8/8 |

Read with care: 8 patients from one lab study. These are descriptive statistics, not a confirmatory test, and nothing here is a clinical claim.
