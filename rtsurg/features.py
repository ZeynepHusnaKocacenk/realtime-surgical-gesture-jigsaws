"""Causal spectral features for freezing-of-gait detection.

The Freeze Index works because freezing shows up as a shift of leg-acceleration power from the
locomotor band (0.5 to 3 Hz) into the "trembling" band (3 to 8 Hz). A small network trained on raw
accelerations from 7 patients has to rediscover this from little data. Giving it the band powers
directly is a cheap, well-motivated prior.

For every frame t and every accelerometer axis, over the last ``window_s`` seconds only (causal):
  * log power in the locomotor band and in the freeze band,
and for the ankle vertical axis also log(Freeze Index). Frames before the first full window are
computed on a zero-padded window, so the output has the same length as the input and never uses
future samples. The cost is one short FFT per axis per frame (64 points at 32 Hz).
"""
from __future__ import annotations

import numpy as np

from .daphnet import ANKLE_VERTICAL

LOCO_BAND, FREEZE_BAND = (0.5, 3.0), (3.0, 8.0)


def spectral_features(x: np.ndarray, rate_hz: float, window_s: float = 2.0) -> np.ndarray:
    """x: [T, F] raw accelerations -> [T, 2F + 1] causal log band powers (+ log FI of the ankle)."""
    x = np.asarray(x, dtype=np.float64)
    T, F = x.shape
    n = int(round(window_s * rate_hz))
    freqs = np.fft.rfftfreq(n, 1.0 / rate_hz)
    loco = (freqs > LOCO_BAND[0]) & (freqs <= LOCO_BAND[1])
    frz = (freqs > FREEZE_BAND[0]) & (freqs <= FREEZE_BAND[1])
    xp = np.concatenate([np.repeat(x[:1], n - 1, 0), x], 0)       # pad with the first sample (no future)
    win = np.lib.stride_tricks.sliding_window_view(xp, n, axis=0)  # [T, F, n]
    out = np.empty((T, 2 * F + 1), np.float32)
    for s in range(0, T, 8192):
        w = win[s:s + 8192]
        p = np.abs(np.fft.rfft(w - w.mean(-1, keepdims=True), axis=-1)) ** 2 / n
        pl, pf = p[..., loco].sum(-1), p[..., frz].sum(-1)            # [t, F]
        out[s:s + len(w), :F] = np.log(pl + 1e-6)
        out[s:s + len(w), F:2 * F] = np.log(pf + 1e-6)
        out[s:s + len(w), 2 * F] = np.log((pf[:, ANKLE_VERTICAL] + 1e-6) / (pl[:, ANKLE_VERTICAL] + 1e-6))
    return out


def featurize(x: np.ndarray, kind: str, rate_hz: float) -> np.ndarray:
    """Model input for a raw trial: ``raw`` = accelerations only, ``spectral`` = raw + band powers."""
    if kind == "raw":
        return np.asarray(x, np.float32)
    if kind == "spectral":
        return np.concatenate([np.asarray(x, np.float32), spectral_features(x, rate_hz)], 1)
    raise ValueError(f"unknown feature set {kind!r}")
