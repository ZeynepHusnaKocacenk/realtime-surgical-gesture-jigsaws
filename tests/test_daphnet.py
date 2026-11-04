"""Tests for the Daphnet loader, splits, event metrics and the Freeze Index.
The fixtures below only exercise the code paths (file format, label handling); no result in the
repository comes from them."""
import numpy as np
import pytest

from rtsurg.daphnet import SAMPLING_HZ, freeze_index, has_freeze, load_daphnet, loso_splits
from rtsurg.fog import (average_precision, event_counts, select_threshold, summarize_counts)


def _write(path, ann, seed=0):
    rng = np.random.default_rng(seed)
    n = len(ann)
    raw = np.column_stack([np.arange(n) * 15.6, rng.normal(size=(n, 9)), ann])
    np.savetxt(path, raw, fmt="%.4f")


@pytest.fixture(scope="module")
def ddir(tmp_path_factory):
    d = tmp_path_factory.mktemp("daphnet")
    # freezes: S01, S02, S03 ; none: S04. Recording S01R01 has an unlabelled gap in the middle.
    a = np.r_[np.full(300, 1), np.full(100, 2), np.full(200, 0), np.full(300, 1), np.full(120, 2), np.full(100, 1)]
    _write(d / "S01R01.txt", a)
    for s in ("S02", "S03"):
        _write(d / f"{s}R01.txt", np.r_[np.full(400, 1), np.full(150, 2), np.full(300, 1)], seed=3)
    _write(d / "S04R01.txt", np.full(800, 1))
    return d


def test_loader_drops_unlabelled_frames_and_splits_at_gaps(ddir):
    data = load_daphnet(ddir, downsample=2)
    s01 = [t for t in data.trials if t.user == "S01"]
    assert len(s01) == 2                                   # split where annotation == 0
    assert sum(len(t.gestures) for t in s01) == (400 + 520) // 2   # 200 unlabelled frames removed
    assert data.n_features == 9 and data.gesture_names == ["no_freeze", "freeze"]
    for t in data.trials:
        assert set(np.unique(t.gestures)) <= {0, 1}
        assert t.kinematics.shape[0] == t.gestures.shape[0]


def test_loader_downsample_averages_samples(ddir):
    d1 = load_daphnet(ddir, downsample=1)
    d2 = load_daphnet(ddir, downsample=2)
    a, b = d1.trials[0].kinematics, d2.trials[0].kinematics
    assert np.allclose(b[0], a[:2].mean(0), atol=1e-4)


def test_loader_missing_dir_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_daphnet(tmp_path)


def test_loso_splits_skip_subjects_without_freeze_and_do_not_leak(ddir):
    data = load_daphnet(ddir)
    folds = list(loso_splits(data, n_val=1, seed=0))
    assert {u for u, *_ in folds} == {"S01", "S02", "S03"}      # S04 has no freeze episodes
    for user, train, val, test in folds:
        tr, va, te = ({t.user for t in x} for x in (train, val, test))
        assert te == {user}
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert has_freeze(val) and "S04" in tr


def test_event_counts_and_always_on_detector_is_not_rewarded():
    t = np.zeros(1000, int); t[200:300] = 1; t[600:650] = 1
    perfect = event_counts([t.copy()], [t], 32.0)
    s = summarize_counts(perfect, 32.0)
    assert s["event_f1"] == 100.0 and s["median_delay_ms"] == 0.0
    always_on = summarize_counts(event_counts([np.ones(1000, int)], [t], 32.0), 32.0)
    assert always_on["event_recall"] == 100.0
    assert always_on["event_precision"] == 0.0 and always_on["specificity"] == 0.0
    late = np.zeros(1000, int); late[216:300] = 1                 # 16 frames late = 500 ms
    c = event_counts([late], [t], 32.0)
    assert c["delays_ms"] == [500.0] and c["detected"] == 1 and c["n_true"] == 2


def test_threshold_selection_uses_only_given_data():
    rng = np.random.default_rng(0)
    y = np.zeros(2000, int); y[500:700] = 1; y[1200:1300] = 1
    p = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.05, 2000), 0, 1)
    thr = select_threshold([p], [y], 32.0)
    assert 0.3 <= thr <= 0.75
    assert average_precision(p, y) > 0.95
    assert 0.0 <= average_precision(rng.random(2000), y) < 0.5


def test_freeze_index_separates_bands_and_is_causal():
    fs = SAMPLING_HZ / 2
    t = np.arange(int(20 * fs)) / fs
    walk, shake = np.sin(2 * np.pi * 1.5 * t), np.sin(2 * np.pi * 5.5 * t)
    fi_w, p_w = freeze_index(walk, fs)
    fi_s, p_s = freeze_index(shake, fs)
    n = int(4 * fs)
    assert fi_w[n:].max() < 0.1 and fi_s[n:].min() > 10          # locomotor vs freeze band
    assert (fi_w[: n - 1] == 0).all() and p_w[n:].min() > 0
    x2 = walk.copy(); x2[300:] = shake[300:]
    fi2, _ = freeze_index(x2, fs)
    assert np.allclose(fi2[:300], fi_w[:300])                    # future samples never change the past
    assert fi2[-1] > 10
