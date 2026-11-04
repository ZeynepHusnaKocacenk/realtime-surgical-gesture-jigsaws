"""Offline vs. online (real-time) evaluation and latency profiling.

Offline protocol : the model sees the whole trial, then labels every frame (literature default).
Online protocol  : at time t the system may only use frames 0..t.
    * causal models   -> their offline output already satisfies this (verified exactly by
                         tests against the frame-by-frame streaming API);
    * acausal models  -> must be re-run on the history x[0..t]; we re-run every
                         ``recompute_stride`` frames and hold the last answer in between,
                         which is what a deployed system would have to do.
"""
from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np
import torch

from .metrics import detection_delay, gesture_metrics, skill_accuracy
from .models import SeqModel


# ---------------------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------------------
@torch.no_grad()
def offline_predictions(model: SeqModel, dataset, device: str):
    g_preds, g_true, s_pred, s_true, s_prefix_logits = [], [], [], [], []
    for x, y, skill, _ in dataset:
        g, s = model(x.unsqueeze(0).to(device))
        g_preds.append(g[0].argmax(-1).cpu().numpy())
        g_true.append(y.numpy())
        s_pred.append(int(s[0, -1].argmax()))
        s_true.append(skill)
        s_prefix_logits.append(s[0].cpu())
    return g_preds, g_true, s_pred, s_true, s_prefix_logits


@torch.no_grad()
def online_predictions_acausal(model: SeqModel, x: torch.Tensor, device: str, stride: int) -> np.ndarray:
    T = len(x)
    pred = np.zeros(T, dtype=np.int64)
    x = x.to(device)
    for t in range(0, T, stride):
        g, _ = model(x[: t + 1].unsqueeze(0))
        pred[t: t + stride] = int(g[0, -1].argmax())
    return pred


@torch.no_grad()
def early_skill_accuracy(model: SeqModel, dataset, device: str, fractions: Sequence[float],
                         offline_prefix_logits=None) -> Dict[float, float]:
    """Skill accuracy when only the first p% of the trial has been observed."""
    out = {}
    for p in fractions:
        preds, trues = [], []
        for i, (x, _, skill, _) in enumerate(dataset):
            k = max(1, int(round(p * len(x))))
            if model.causal and offline_prefix_logits is not None:
                logit = offline_prefix_logits[i][k - 1]
            else:
                _, s = model(x[:k].unsqueeze(0).to(device))
                logit = s[0, -1]
            preds.append(int(logit.argmax())); trues.append(skill)
        out[p] = skill_accuracy(preds, trues)
    return out


def evaluate_fold(model: SeqModel, dataset, device: str, frame_ms: float,
                  recompute_stride: int, fractions: Sequence[float]) -> Dict:
    model.eval()
    g_off, g_true, s_pred, s_true, prefix = offline_predictions(model, dataset, device)
    if model.causal:
        g_on = g_off                                    # identical by construction (see tests)
    else:
        g_on = [online_predictions_acausal(model, x, device, recompute_stride) for x, *_ in dataset]
    delays, hit, total = [], 0, 0
    for p, t in zip(g_on, g_true):
        d, h, n = detection_delay(p, t, frame_ms)
        delays += d; hit += h; total += n
    return {
        "offline": gesture_metrics(g_off, g_true),
        "online": gesture_metrics(g_on, g_true),
        "skill_acc": skill_accuracy(s_pred, s_true),
        "early_skill": early_skill_accuracy(model, dataset, device, fractions, prefix),
        "delay_ms": delays,
        "detected_frac": hit / max(total, 1),
        "example": {"true": g_true[0].tolist(), "online": g_on[0].tolist(), "offline": g_off[0].tolist()},
    }


# ---------------------------------------------------------------------------------------
# Latency profiling
# ---------------------------------------------------------------------------------------
class _Timer:
    """Wall-clock timer in ms; uses CUDA events on GPU so asynchronous kernels are counted."""

    def __init__(self, device: str):
        self.cuda = device.startswith("cuda")

    def time(self, fn) -> float:
        if self.cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); fn(); end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        t0 = time.perf_counter(); fn()
        return (time.perf_counter() - t0) * 1000.0


@torch.no_grad()
def profile_latency(model: SeqModel, n_features: int, device: str, history_frames: Sequence[int],
                    n_iter: int = 200, warmup: int = 20) -> Dict:
    """Per-update latency in ms (batch size 1, as in a live system).

    * causal   -> one ``stream_step`` (cost does not depend on how long surgery has run);
    * acausal  -> one full re-run over the history, measured at several history lengths.
    """
    model.eval().to(device)
    timer = _Timer(device)
    res: Dict = {"by_history": {}}
    if model.causal:
        model.stream_reset()
        frames = torch.randn(warmup + n_iter, n_features, device=device)
        for i in range(warmup):
            model.stream_step(frames[i])
        ts = [timer.time(lambda f=frames[warmup + i]: model.stream_step(f)) for i in range(n_iter)]
        stats = _stats(ts)
        for h in history_frames:
            res["by_history"][int(h)] = stats
    else:
        for h in history_frames:
            x = torch.randn(1, int(h), n_features, device=device)
            for _ in range(max(3, warmup // 4)):
                model(x)
            n = max(10, n_iter // 10)
            res["by_history"][int(h)] = _stats([timer.time(lambda: model(x)) for _ in range(n)])
    return res


def _stats(ts: List[float]) -> Dict[str, float]:
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}
