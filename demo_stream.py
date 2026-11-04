"""Streaming demo: train a small causal TCN on synthetic data, then stream one held-out trial
frame by frame with ``stream_step`` and report per-frame latency against the 100 ms budget.

    python demo_stream.py

The data is generated, so the numbers are a pipeline demo and not a result.
"""
from __future__ import annotations

import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from rtsurg.data import (FRAME_RATE_HZ, Normalizer, TrialDataset, collate_padded,
                         generate_synthetic_jigsaws, load_jigsaws_task, louo_splits)
from rtsurg.models import build_model
from rtsurg.train import TrainConfig, train_model

BANNER = "SYNTHETIC DATA, pipeline demo, not a result"
BUDGET_MS = 100.0
DOWNSAMPLE = 3                                    # 30 Hz / 3 = 10 Hz -> 100 ms per frame


def main(epochs: int = 8, seed: int = 0, max_print: int = 25):
    print(f"=== {BANNER} ===")
    torch.manual_seed(seed); np.random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        data = load_jigsaws_task(generate_synthetic_jigsaws(tmp, trials_per_user=3, seed=seed))
    user, train_trials, test_trials = next(iter(louo_splits(data)))
    norm = Normalizer.fit(train_trials)
    train_ds = TrialDataset(train_trials, norm, DOWNSAMPLE)
    test_ds = TrialDataset(test_trials, norm, DOWNSAMPLE)
    dl = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_padded)
    model = build_model("TCN-causal", data.n_features, data.n_gestures, hidden=32, tcn_layers=6)
    cfg = TrainConfig(epochs=epochs, device="cpu", seed=seed)
    print(f"training a causal TCN for {epochs} epochs (held-out surgeon: {user}) ...")
    train_model(model, dl, dl, cfg)

    x, y, _, _ = test_ds[0]
    model.eval(); model.stream_reset()
    lat, preds = [], []
    print(f"\nstreaming one held-out trial, {len(x)} frames at {FRAME_RATE_HZ / DOWNSAMPLE:.0f} Hz")
    print(f"{'frame':>6} {'pred':>5} {'true':>5} {'latency ms':>11}")
    for t in range(len(x)):
        t0 = time.perf_counter()
        g, _ = model.stream_step(x[t])
        p = int(g.argmax())
        ms = (time.perf_counter() - t0) * 1000.0
        lat.append(ms); preds.append(p)
        if t < max_print:
            print(f"{t:>6} {p:>5} {int(y[t]):>5} {ms:>11.3f}")
    if len(x) > max_print:
        print(f"   ... ({len(x) - max_print} more frames)")
    within = sum(l <= BUDGET_MS for l in lat)
    acc = 100.0 * float(np.mean(np.array(preds) == y.numpy()))
    print(f"\nframe accuracy on this trial: {acc:.1f}%")
    print(f"latency median {np.median(lat):.3f} ms, p95 {np.percentile(lat, 95):.3f} ms")
    print(f"frames within the {BUDGET_MS:.0f} ms budget: {within}/{len(lat)}")
    print(f"=== {BANNER} ===")


if __name__ == "__main__":
    main()
