# Validation-only hyperparameter search (causal TCN, Daphnet)

16 configurations x 8 leave-one-subject-out folds (seed 0). Each configuration is scored on the **validation patients** of each fold only; the test patients were never used. Ranked by mean validation PR-AUC (the selection criterion); validation event F1 is shown for reference. Both are optimistic (epoch and threshold are also chosen on the validation patients) and are not performance estimates.

**Selected:** `spectral_h64_L10_lr0.002_do0.3_w256` (mean validation PR-AUC 0.647). It is then evaluated on the test patients with 3 seeds (`configs/daphnet_tuned.yaml`).

| Rank | Features | Hidden | Layers | Receptive field (s) | LR | Dropout | Window (s) | Params | Val PR-AUC | Val event F1 | CPU min |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | spectral | 64 | 10 | 64.0 | 0.002 | 0.3 | 8 | 171,461 | 0.647 | 66.0 | 57 |
| 2 | spectral | 32 | 8 | 16.0 | 0.002 | 0.3 | 4 | 35,429 | 0.640 | 68.3 | 16 |
| 3 | spectral | 32 | 8 | 16.0 | 0.002 | 0.3 | 16 | 35,429 | 0.639 | 65.8 | 13 |
| 4 | spectral | 32 | 8 | 16.0 | 0.001 | 0.3 | 8 | 35,429 | 0.639 | 67.4 | 14 |
| 5 | spectral | 32 | 8 | 16.0 | 0.002 | 0.1 | 8 | 35,429 | 0.638 | 66.0 | 13 |
| 6 | spectral | 32 | 10 | 64.0 | 0.002 | 0.3 | 8 | 43,749 | 0.638 | 66.6 | 28 |
| 7 | spectral | 32 | 6 | 4.0 | 0.002 | 0.3 | 8 | 27,109 | 0.638 | 68.1 | 10 |
| 8 | spectral | 64 | 6 | 4.0 | 0.002 | 0.3 | 8 | 105,413 | 0.637 | 66.6 | 37 |
| 9 | spectral | 32 | 8 | 16.0 | 0.002 | 0.3 | 8 | 35,429 | 0.635 | 66.6 | 14 |
| 10 | spectral | 32 | 8 | 16.0 | 0.002 | 0.5 | 8 | 35,429 | 0.633 | 67.1 | 15 |
| 11 | spectral | 64 | 8 | 16.0 | 0.002 | 0.3 | 8 | 138,437 | 0.631 | 65.6 | 45 |
| 12 | spectral | 16 | 10 | 64.0 | 0.002 | 0.3 | 8 | 11,381 | 0.630 | 61.6 | 9 |
| 13 | spectral | 32 | 8 | 16.0 | 0.005 | 0.3 | 8 | 35,429 | 0.629 | 66.2 | 14 |
| 14 | spectral | 16 | 8 | 16.0 | 0.002 | 0.3 | 8 | 9,269 | 0.619 | 63.2 | 7 |
| 15 | spectral | 16 | 6 | 4.0 | 0.002 | 0.3 | 8 | 7,157 | 0.617 | 63.7 | 5 |
| 16 | raw | 32 | 8 | 16.0 | 0.002 | 0.3 | 8 | 34,821 | 0.460 | 57.7 | 14 |
