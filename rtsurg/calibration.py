"""Temperature scaling and calibration metrics. The temperature is fitted on VALIDATION
subjects only; it never changes the argmax, only the confidence."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, T: float = 1.0) -> float:
    p = softmax(logits, T)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """T > 0 minimising the negative log-likelihood of ``labels`` (1-D search on log T, refined)."""
    lo, hi = np.log(0.05), np.log(20.0)
    for _ in range(3):                                   # coarse-to-fine grid
        grid = np.linspace(lo, hi, 41)
        vals = [nll(logits, labels, float(np.exp(g))) for g in grid]
        i = int(np.argmin(vals))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    return float(np.exp(grid[i]))


def ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (equal-width bins): sum_b |acc_b - conf_b| * n_b / n."""
    confidence, correct = np.asarray(confidence, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def calibration_report(logits: np.ndarray, labels: np.ndarray, T: float) -> dict:
    """ECE of the top-class confidence and of the freeze probability, before (T=1) and after."""
    out = {}
    for tag, t in (("before", 1.0), ("after", T)):
        p = softmax(logits, t)
        conf, pred = p.max(-1), p.argmax(-1)
        out[f"ece_{tag}"] = ece(conf, pred == labels)
        out[f"freeze_ece_{tag}"] = ece(p[:, 1], labels == 1)
        out[f"nll_{tag}"] = nll(logits, labels, t)
    return out
