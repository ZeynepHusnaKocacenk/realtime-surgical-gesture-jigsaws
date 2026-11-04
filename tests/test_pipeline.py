"""Correctness tests: data parsing, LOUO leakage, metric sanity, and the key real-time
guarantee that frame-by-frame streaming reproduces the offline causal outputs."""
import numpy as np
import pytest
import torch

from rtsurg.data import (Normalizer, TrialDataset, generate_synthetic_jigsaws, load_jigsaws_task,
                         louo_splits)
from rtsurg.metrics import edit_score, f1_at_k, frame_accuracy, segments
from rtsurg.models import MODEL_NAMES, build_model


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = generate_synthetic_jigsaws(tmp_path_factory.mktemp("jig"), trials_per_user=2, seed=1)
    return load_jigsaws_task(d)


def test_loader_shapes(data):
    assert len(data.trials) == 16
    assert data.n_features == 38                       # PSM_L + PSM_R
    for t in data.trials:
        assert len(t.kinematics) == len(t.gestures) > 0
        assert t.skill in (0, 1, 2)


def test_louo_no_leakage(data):
    folds = list(louo_splits(data))
    assert len(folds) == len(data.users)
    for user, train, test in folds:
        assert all(t.user == user for t in test)
        assert all(t.user != user for t in train)


def test_metrics_sanity():
    y = np.array([0, 0, 1, 1, 1, 2, 2])
    assert frame_accuracy(y, y) == 100.0
    assert edit_score(y, y) == 100.0
    assert f1_at_k(y, y, 0.5) == 100.0
    assert segments(y) == [(0, 0, 2), (1, 2, 5), (2, 5, 7)]
    over = np.array([0, 1, 0, 1, 1, 2, 2])
    assert edit_score(over, y) < 100.0


@pytest.mark.parametrize("name", ["TCN-causal", "LSTM"])
def test_streaming_matches_offline(name):
    torch.manual_seed(0)
    model = build_model(name, n_features=10, n_gestures=5, hidden=16, tcn_layers=4).eval()
    x = torch.randn(1, 60, 10)                          # longer than the TCN receptive field (31)
    with torch.no_grad():
        g_off, s_off = model(x)
        model.stream_reset()
        steps = [model.stream_step(x[0, t]) for t in range(60)]
    g_on = torch.stack([g for g, _ in steps])
    s_on = torch.stack([s for _, s in steps])
    assert torch.allclose(g_off[0], g_on, atol=1e-5)
    assert torch.allclose(s_off[0], s_on, atol=1e-5)


@pytest.mark.parametrize("name", ["TCN-causal", "LSTM"])
def test_causal_models_ignore_future(name):
    torch.manual_seed(0)
    model = build_model(name, n_features=6, n_gestures=4, hidden=8, tcn_layers=3).eval()
    x = torch.randn(1, 40, 6)
    x2 = x.clone(); x2[:, 25:] = torch.randn(1, 15, 6)
    with torch.no_grad():
        a, _ = model(x); b, _ = model(x2)
    assert torch.allclose(a[:, :25], b[:, :25], atol=1e-6)


@pytest.mark.parametrize("name", ["TCN-acausal", "BiLSTM"])
def test_acausal_models_use_future(name):
    torch.manual_seed(0)
    model = build_model(name, n_features=6, n_gestures=4, hidden=8, tcn_layers=3).eval()
    x = torch.randn(1, 40, 6)
    x2 = x.clone(); x2[:, 25:] = torch.randn(1, 15, 6)
    with torch.no_grad():
        a, _ = model(x); b, _ = model(x2)
    assert not torch.allclose(a[:, :25], b[:, :25], atol=1e-6)


def test_padding_does_not_change_outputs(data):
    norm = Normalizer.fit(data.trials)
    ds = TrialDataset(data.trials[:2], norm)
    from rtsurg.data import collate_padded
    X, Y, S, M, _ = collate_padded([ds[0], ds[1]])
    for name in MODEL_NAMES:
        torch.manual_seed(0)
        model = build_model(name, data.n_features, data.n_gestures, hidden=8, tcn_layers=3).eval()
        with torch.no_grad():
            g_b, _ = model(X, M)
            for i in range(2):
                g_i, _ = model(ds[i][0].unsqueeze(0))
                n = len(ds[i][0])
                if model.causal or n == X.shape[1]:
                    assert torch.allclose(g_b[i, :n], g_i[0], atol=1e-4), name
