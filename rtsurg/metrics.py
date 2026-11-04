"""Evaluation metrics used in the surgical gesture recognition literature.

* frame accuracy      : % of frames labelled correctly
* edit score          : segmental Levenshtein score (Lea et al., 2016), penalises over-segmentation
* F1@{10,25,50}       : segmental F1 at IoU thresholds (Lea et al., 2017; Farha & Gall, MS-TCN 2019)
* detection delay     : real-time metric, how long after a gesture starts the online model
                        first outputs the correct label
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def segments(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Run-length encode -> list of (label, start, end_exclusive)."""
    labels = np.asarray(labels)
    if len(labels) == 0:
        return []
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(labels)]])
    return [(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def frame_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean() * 100.0)


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    d = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int64)
    d[:, 0] = np.arange(len(a) + 1)
    d[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(d[-1, -1])


def edit_score(pred: np.ndarray, true: np.ndarray) -> float:
    p = [s[0] for s in segments(pred)]
    t = [s[0] for s in segments(true)]
    return float((1.0 - _levenshtein(p, t) / max(len(p), len(t), 1)) * 100.0)


def f1_at_k(pred: np.ndarray, true: np.ndarray, iou: float) -> float:
    ps, ts = segments(pred), segments(true)
    used = np.zeros(len(ts), dtype=bool)
    tp = 0
    for lab, s, e in ps:
        best, best_j = 0.0, -1
        for j, (tl, ts_, te) in enumerate(ts):
            if tl != lab:
                continue
            inter = max(0, min(e, te) - max(s, ts_))
            union = max(e, te) - min(s, ts_)
            score = inter / union if union else 0.0
            if score > best:
                best, best_j = score, j
        if best >= iou and not used[best_j]:
            tp += 1
            used[best_j] = True
    fp, fn = len(ps) - tp, len(ts) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * prec * rec / (prec + rec) * 100.0) if prec + rec else 0.0


def gesture_metrics(preds: List[np.ndarray], trues: List[np.ndarray]) -> Dict[str, float]:
    """Frame accuracy pooled over all frames; segmental scores averaged over trials."""
    allp, allt = np.concatenate(preds), np.concatenate(trues)
    return {
        "frame_acc": frame_accuracy(allp, allt),
        "edit": float(np.mean([edit_score(p, t) for p, t in zip(preds, trues)])),
        "f1@10": float(np.mean([f1_at_k(p, t, 0.10) for p, t in zip(preds, trues)])),
        "f1@25": float(np.mean([f1_at_k(p, t, 0.25) for p, t in zip(preds, trues)])),
        "f1@50": float(np.mean([f1_at_k(p, t, 0.50) for p, t in zip(preds, trues)])),
    }


def detection_delay(pred: np.ndarray, true: np.ndarray, frame_ms: float) -> Tuple[List[float], int, int]:
    """For every true gesture segment, delay (ms) from its onset until the online prediction
    first equals the true label inside that segment. Returns (delays, n_detected, n_segments)."""
    delays, total = [], 0
    for lab, s, e in segments(true):
        total += 1
        hit = np.flatnonzero(pred[s:e] == lab)
        if len(hit):
            delays.append(float(hit[0] * frame_ms))
    return delays, len(delays), total


def skill_accuracy(pred: Sequence[int], true: Sequence[int]) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    return float((pred == true).mean() * 100.0) if len(true) else float("nan")
