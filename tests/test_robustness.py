"""Tests for the fault injectors and temperature scaling (no dataset needed)."""
import numpy as np

from rtsurg.calibration import ece, fit_temperature, nll, softmax
from rtsurg.robustness import add_noise, apply_fault, drop_frames, sensor_freeze


def _x(n=2000, f=4, seed=0):
    return np.random.default_rng(seed).normal(size=(n, f)).astype(np.float32)


def test_drop_frames_holds_last_received_frame():
    x = _x(); rng = np.random.default_rng(1)
    y = drop_frames(x, 0.2, rng)
    assert y.shape == x.shape and np.array_equal(y[0], x[0])
    changed = np.flatnonzero((y != x).any(1))
    assert len(changed) > 0
    for t in changed:                       # every altered frame equals the previous output frame
        assert np.array_equal(y[t], y[t - 1])
    assert abs((~(y == x).all(1)).mean() - 0.2) < 0.05
    assert np.array_equal(drop_frames(x, 0.0, rng), x)
    assert np.array_equal(drop_frames(x, 0.2, np.random.default_rng(1)), y)     # deterministic


def test_noise_scales_with_feature_std():
    x = np.zeros((20000, 3), np.float32)
    std = np.array([1.0, 5.0, 0.1])
    y = add_noise(x, 0.5, std, np.random.default_rng(0))
    assert np.allclose(y.std(0), 0.5 * std, rtol=0.05)
    assert y.dtype == x.dtype and np.array_equal(add_noise(x, 0.0, std, np.random.default_rng(0)), x)


def test_sensor_freeze_converts_seconds_to_frames():
    rate = 32.0
    x = np.arange(4000, dtype=np.float32)[:, None] * np.ones((1, 2), np.float32)
    y = sensor_freeze(x, 2.0, rate, np.random.default_rng(3))
    changed = np.flatnonzero((y != x).any(1))
    n = int(round(2.0 * rate))
    assert len(changed) > 0 and len(changed) <= (4000 // (30 * 32) + 1) * n
    runs = np.split(changed, np.flatnonzero(np.diff(changed) > 1) + 1)
    assert max(len(r) for r in runs) <= n
    assert np.array_equal(apply_fault(x, "clean", 0, np.ones(2), rate, np.random.default_rng(0)), x)
    assert not np.array_equal(apply_fault(x, "freeze", 3, np.ones(2), rate, np.random.default_rng(0)), x)


def test_temperature_scaling_fixes_overconfidence_and_keeps_argmax():
    rng = np.random.default_rng(0)
    n = 20000
    true_logits = rng.normal(size=(n, 2)) * 1.5
    y = (rng.random(n) < softmax(true_logits)[:, 1]).astype(int)       # labels drawn from the true probs
    over = true_logits * 3.0                                            # model is 3x over-confident
    T = fit_temperature(over, y)
    assert 2.5 < T < 3.5
    p_before, p_after = softmax(over), softmax(over, T)
    assert np.array_equal(p_before.argmax(1), p_after.argmax(1))
    e0 = ece(p_before.max(1), p_before.argmax(1) == y)
    e1 = ece(p_after.max(1), p_after.argmax(1) == y)
    assert e1 < e0 and e1 < 0.02
    assert nll(over, y, T) < nll(over, y, 1.0)


def test_ece_is_zero_for_perfect_calibration_and_large_when_wrong():
    conf = np.full(1000, 0.8); correct = np.r_[np.ones(800), np.zeros(200)]
    assert ece(conf, correct) < 1e-9
    assert ece(np.full(1000, 0.99), correct) > 0.15
