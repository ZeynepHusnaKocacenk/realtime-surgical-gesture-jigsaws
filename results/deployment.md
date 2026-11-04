# Deployment benchmark (design proposal, not a clinical claim)

CPU, 1 thread, batch size 1, one streaming step per frame; frame budget 31.2 ms (32 Hz). Exported graph = one step (new frame + model state in, freeze logits + new state out). Outputs of the exported fp32 models match the offline PyTorch pass (max abs difference below). Accuracy change is measured on all 8 Daphnet test subjects (seed 0) with the decision threshold chosen on validation subjects using the fp32 ONNX model and then applied unchanged to int8.

## TCN-causal (26,501 parameters)

| Format | Median ms | p95 ms | Size (MB) | Max abs diff vs offline | Event F1 | Event recall | PR-AUC |
|---|---|---|---|---|---|---|---|
| PyTorch (eager) | 0.412 | 0.575 | 0.118 | 1.9e-06 | n/a | n/a | n/a |
| TorchScript | 0.096 | 0.144 | 0.119 | 1.4e-06 | n/a | n/a | n/a |
| ONNX Runtime fp32 | 0.042 | 0.069 | 0.115 | 1.9e-06 | 52.6 | 67.5 | 0.517 |
| ONNX Runtime int8 (dynamic) | 0.084 | 0.152 | 0.064 | 5.7e-02 | 52.4 | 67.5 | 0.517 |

Event F1 change after int8 quantisation: -0.2 points.

## LSTM (11,045 parameters)

| Format | Median ms | p95 ms | Size (MB) | Max abs diff vs offline | Event F1 | Event recall | PR-AUC |
|---|---|---|---|---|---|---|---|
| PyTorch (eager) | 0.158 | 0.253 | 0.050 | 1.4e-06 | n/a | n/a | n/a |
| TorchScript | 0.104 | 0.169 | 0.051 | 1.4e-06 | n/a | n/a | n/a |
| ONNX Runtime fp32 | 0.016 | 0.027 | 0.041 | 9.5e-07 | 29.0 | 55.7 | 0.360 |
| ONNX Runtime int8 (dynamic) | 0.023 | 0.041 | 0.016 | 1.5e-02 | 29.2 | 55.3 | 0.357 |

Event F1 change after int8 quantisation: +0.2 points.

