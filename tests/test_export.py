"""Exported streaming-step models must reproduce the PyTorch outputs."""
import numpy as np
import pytest
import torch

pytest.importorskip("onnx"); pytest.importorskip("onnxruntime")
from export import Runner, export_all, make_step, stream_logits          # noqa: E402
from rtsurg.models import build_model                                     # noqa: E402


@pytest.mark.parametrize("name", ["TCN-causal", "LSTM"])
def test_exports_match_pytorch(name, tmp_path):
    torch.manual_seed(0)
    model = build_model(name, n_features=9, n_gestures=2, hidden=16, tcn_layers=4).eval()
    step = make_step(model)
    paths = export_all(model, step, tmp_path, name)
    x = np.random.default_rng(0).normal(size=(120, 9)).astype(np.float32)   # longer than the receptive field
    with torch.no_grad():
        ref = model(torch.from_numpy(x).unsqueeze(0))[0][0].numpy()
    for kind in ("pytorch", "torchscript", "onnx"):
        out = stream_logits(Runner(kind, model, step, paths), x)
        assert np.abs(out - ref).max() < 1e-4, kind
    q = stream_logits(Runner("onnx_int8", model, step, paths), x)            # int8: close, not identical
    assert q.shape == ref.shape and np.isfinite(q).all()
    assert (q.argmax(1) == ref.argmax(1)).mean() > 0.8
