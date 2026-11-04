"""Sensor-fault injectors and the robustness / abstention evaluation for the Daphnet benchmark.

Faults are applied to the RAW (un-normalised) test signals, then the model's normaliser is applied,
exactly as a live system would see them. All injectors are deterministic given a numpy Generator.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

CONDITIONS = ([("clean", 0.0)] + [("drop", p) for p in (0.05, 0.10, 0.20)]
              + [("noise", s) for s in (0.25, 0.5, 1.0)] + [("freeze", d) for d in (1, 2, 3)])
LABELS = {"clean": "clean", "drop": "dropped frames", "noise": "Gaussian noise (x feature std)",
          "freeze": "sensor freeze (s)"}


def cond_name(kind: str, level: float) -> str:
    return "clean" if kind == "clean" else {"drop": f"drop {level * 100:.0f}%", "noise": f"noise {level}x std",
                                            "freeze": f"stuck {level:.0f} s"}[kind]


def drop_frames(x: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Each frame is lost with probability p; the last received frame is held. Frame 0 is kept."""
    lost = rng.random(len(x)) < p
    lost[0] = False
    idx = np.where(lost, 0, np.arange(len(x)))
    return x[np.maximum.accumulate(idx)]


def add_noise(x: np.ndarray, rel_sigma: float, feature_std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with std = rel_sigma * (per-feature std computed on TRAINING data)."""
    return (x + rng.normal(size=x.shape) * (rel_sigma * feature_std)).astype(x.dtype)


def sensor_freeze(x: np.ndarray, seconds: float, rate_hz: float, rng: np.random.Generator,
                  every_s: float = 30.0) -> np.ndarray:
    """Stuck sensor: the output stops updating for ``seconds``. One event per ``every_s`` of
    recording (at least one), at random start times; the value at the start is held."""
    out, n = x.copy(), max(int(round(seconds * rate_hz)), 1)
    n_events = max(1, int(len(x) / (every_s * rate_hz)))
    for s in rng.integers(0, max(len(x) - n, 1), size=n_events):
        out[s:s + n] = out[s]
    return out


def apply_fault(x: np.ndarray, kind: str, level: float, feature_std: np.ndarray, rate_hz: float,
                rng: np.random.Generator) -> np.ndarray:
    if kind == "clean":
        return x
    if kind == "drop":
        return drop_frames(x, level, rng)
    if kind == "noise":
        return add_noise(x, level, feature_std, rng)
    if kind == "freeze":
        return sensor_freeze(x, level, rate_hz, rng)
    raise ValueError(kind)
