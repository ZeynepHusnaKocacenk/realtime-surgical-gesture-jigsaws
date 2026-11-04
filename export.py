"""Export trained causal Daphnet models for deployment, verify them, and benchmark streaming latency.

For each causal model (TCN-causal, LSTM) the exported graph is ONE streaming step: it takes the newest
frame plus the model state (per-layer caches / LSTM hidden state) and returns the freeze logits and the
new state. Formats compared on CPU, one thread, batch size 1:
  PyTorch eager (stream_step) · TorchScript · ONNX Runtime fp32 · ONNX Runtime dynamic int8.

    python export.py --data-dir /path/to/dataset_fog_release/dataset --models-dir results/daphnet/models

Design proposal only: nothing here has been validated for clinical use.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rtsurg.models import LSTMModel, TCN, build_model


# ---------------------------------------------------------------------------------------
# Stateless streaming-step wrappers (state in, state out)
# ---------------------------------------------------------------------------------------
class TCNStep(nn.Module):
    """One TCN frame. State = one cache [(k-1)*d, C] per layer (oldest row first)."""

    def __init__(self, m: TCN):
        super().__init__()
        self.k = [l.kernel for l in m.layers]; self.d = [l.dilation for l in m.layers]
        self.register_buffer("inp_w", m.inp.weight[:, :, 0].t().contiguous())
        self.register_buffer("inp_b", m.inp.bias.detach().clone())
        for i, l in enumerate(m.layers):
            for j in range(l.kernel):
                self.register_buffer(f"w{i}_{j}", l.conv.weight[:, :, j].t().contiguous())
            self.register_buffer(f"b{i}", l.conv.bias.detach().clone())
            self.register_buffer(f"ow{i}", l.out.weight[:, :, 0].t().contiguous())
            self.register_buffer(f"ob{i}", l.out.bias.detach().clone())
        self.register_buffer("hw", m.gesture_head.weight.t().contiguous())
        self.register_buffer("hb", m.gesture_head.bias.detach().clone())
        self.channels = m.inp.out_channels
        self.n_features = m.inp.in_channels

    def init_state(self) -> List[torch.Tensor]:
        return [torch.zeros((k - 1) * d, self.channels) for k, d in zip(self.k, self.d)]

    def forward(self, x, *caches):
        h = x @ self.inp_w + self.inp_b                                   # [1, C]
        new = []
        for i, cache in enumerate(caches):
            k, d = self.k[i], self.d[i]
            y = h @ getattr(self, f"w{i}_{k - 1}") + getattr(self, f"b{i}")
            for j in range(k - 1):
                y = y + cache[j * d: j * d + 1] @ getattr(self, f"w{i}_{j}")
            o = F.relu(y) @ getattr(self, f"ow{i}") + getattr(self, f"ob{i}")
            new.append(torch.cat([cache[1:], h], 0))
            h = h + o
        return (h @ self.hw + self.hb, *new)


class LSTMStep(nn.Module):
    """One LSTM frame. State = (h, c), each [1, 1, H]."""

    def __init__(self, m: LSTMModel):
        super().__init__()
        self.inp, self.lstm, self.proj, self.head = m.inp, m.lstm, m.proj, m.gesture_head
        self.hidden, self.n_features = m.lstm.hidden_size, m.inp.in_features

    def init_state(self):
        return [torch.zeros(1, 1, self.hidden), torch.zeros(1, 1, self.hidden)]

    def forward(self, x, h, c):
        z = F.relu(self.inp(x)).unsqueeze(1)                              # [1, 1, H]
        out, (h2, c2) = self.lstm(z, (h, c))
        return (self.head(self.proj(out)[:, 0]), h2, c2)


def make_step(model) -> nn.Module:
    return (TCNStep(model) if isinstance(model, TCN) else LSTMStep(model)).eval()


def load_model(path: Path):
    ck = torch.load(path, weights_only=False)
    c = ck["cfg"]
    m = build_model(c["model"], c["n_features"], 2, c["hidden"], c["tcn_layers"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


# ---------------------------------------------------------------------------------------
# Export + runners
# ---------------------------------------------------------------------------------------
def export_all(model, step: nn.Module, out_dir: Path, tag: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.zeros(1, step.n_features)
    state = step.init_state()
    names_in = ["x"] + [f"s{i}" for i in range(len(state))]
    names_out = ["logits"] + [f"s{i}_out" for i in range(len(state))]
    paths = {"torchscript": out_dir / f"{tag}.torchscript.pt", "onnx": out_dir / f"{tag}.onnx",
             "onnx_int8": out_dir / f"{tag}.int8.onnx"}
    ts = torch.jit.trace(step, (x, *state), check_trace=False)
    ts.save(str(paths["torchscript"]))
    with torch.no_grad():
        torch.onnx.export(step, (x, *state), str(paths["onnx"]), input_names=names_in, output_names=names_out,
                          opset_version=17, dynamo=False)
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(paths["onnx"]), str(paths["onnx_int8"]), weight_type=QuantType.QInt8)
    return paths


class Runner:
    """Uniform streaming interface: reset(); step(frame[F]) -> logits[2]."""

    def __init__(self, kind: str, model, step: nn.Module, paths: Dict[str, Path]):
        self.kind, self.model, self.step_mod = kind, model, step
        if kind.startswith("onnx"):
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(str(paths[kind]), so, providers=["CPUExecutionProvider"])
            self.in_names = [i.name for i in self.sess.get_inputs()]
            self.out_names = [o.name for o in self.sess.get_outputs()]
        elif kind == "torchscript":
            self.ts = torch.jit.load(str(paths["torchscript"]))
        self.reset()

    def reset(self):
        if self.kind == "pytorch":
            self.model.stream_reset()
        elif self.kind == "torchscript":
            self.state = self.step_mod.init_state()
        else:
            self.state = [s.numpy() for s in self.step_mod.init_state()]

    def step(self, frame: np.ndarray) -> np.ndarray:
        if self.kind == "pytorch":
            g, _ = self.model.stream_step(torch.from_numpy(frame)); return g.numpy()
        if self.kind == "torchscript":
            with torch.no_grad():
                out = self.ts(torch.from_numpy(frame).view(1, -1), *self.state)
            self.state = list(out[1:]); return out[0].numpy()[0]
        feed = dict(zip(self.in_names, [frame.reshape(1, -1).astype(np.float32), *self.state]))
        out = self.sess.run(self.out_names, feed)
        self.state = out[1:]; return out[0][0]


def stream_logits(runner: Runner, x: np.ndarray) -> np.ndarray:
    runner.reset()
    return np.stack([runner.step(f) for f in x])


def latency(runner: Runner, n_features: int, n: int = 1500, warmup: int = 100) -> Dict[str, float]:
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(warmup + n, n_features)).astype(np.float32)
    runner.reset()
    for f in frames[:warmup]:
        runner.step(f)
    ts = []
    for f in frames[warmup:]:
        t0 = time.perf_counter(); runner.step(f); ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(ts)
    return {"median_ms": float(np.median(a)), "p95_ms": float(np.percentile(a, 95)), "mean_ms": float(a.mean())}


def _acc(v, rate):
    from rtsurg.fog import merge_counts, summarize_counts
    sm = summarize_counts(merge_counts([a["counts"] for a in v]), rate)
    return {"event_f1": sm["event_f1"], "recall": sm["event_recall"],
            "pr_auc": float(np.mean([a["pr_auc"] for a in v]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models-dir", default="results/daphnet/models")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", nargs="+", default=["TCN-causal", "LSTM"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--downsample", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(1)
    from rtsurg.daphnet import SAMPLING_HZ, load_daphnet, loso_splits
    from rtsurg.data import Normalizer
    from rtsurg.fog import average_precision, event_counts, freeze_prob, select_threshold, summarize_counts
    out = Path(args.out); dep = out / "deployment"
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample; budget = 1000.0 / rate
    folds = list(loso_splits(data, args.n_val, args.seed))
    res: Dict = {"budget_ms": budget, "models": {}}
    kinds = ["pytorch", "torchscript", "onnx", "onnx_int8"]
    for name in args.models:
        # ---- export fold 0's model (latency, size, equivalence) ---------------------------
        user0 = folds[0][0]
        model, _ = load_model(Path(args.models_dir) / f"{name}_seed{args.seed}_{user0}.pt")
        step = make_step(model)
        paths = export_all(model, step, dep, f"{name}_{user0}")
        runners = {k: Runner(k, model, step, paths) for k in kinds}
        x = np.random.default_rng(1).normal(size=(300, step.n_features)).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x).unsqueeze(0))[0][0].numpy()      # offline pass
        diff = {k: float(np.abs(stream_logits(r, x) - ref).max()) for k, r in runners.items()}
        for k in ("pytorch", "torchscript", "onnx"):
            assert diff[k] < 1e-3, (name, k, diff[k])
        lat = {k: latency(r, step.n_features) for k, r in runners.items()}
        size = {"pytorch": os.path.getsize(Path(args.models_dir) / f"{name}_seed{args.seed}_{user0}.pt") / 1e6,
                "torchscript": paths["torchscript"].stat().st_size / 1e6,
                "onnx": paths["onnx"].stat().st_size / 1e6, "onnx_int8": paths["onnx_int8"].stat().st_size / 1e6}
        # ---- accuracy change after quantisation, all folds (validation threshold) ---------
        acc = {"onnx": [], "onnx_int8": []}
        for user, train, val, test in folds:
            mdl, ck = load_model(Path(args.models_dir) / f"{name}_seed{args.seed}_{user}.pt")
            stp = make_step(mdl); pth = export_all(mdl, stp, dep / "tmp", f"{name}_{user}")
            norm = Normalizer(mean=ck["norm_mean"], std=ck["norm_std"])
            vx = [norm(t.kinematics) for t in val]; vy = [t.gestures for t in val]
            tx = [norm(t.kinematics) for t in test]; ty = [t.gestures for t in test]
            fp32 = Runner("onnx", mdl, stp, pth)
            pv = [freeze_prob(stream_logits(fp32, v)) for v in vx]
            thr = select_threshold(pv, vy, rate)
            for k in ("onnx", "onnx_int8"):
                r = Runner(k, mdl, stp, pth)
                pt = [freeze_prob(stream_logits(r, t)) for t in tx]
                acc[k].append({"fold": user, "counts": event_counts([p >= thr for p in pt], ty, rate),
                               "pr_auc": average_precision(np.concatenate(pt), np.concatenate(ty))})
            print(f"[{name}] fold {user} quantisation check done", flush=True)
        res["models"][name] = {"max_abs_diff_vs_offline": diff, "latency": lat, "size_mb": size,
                               "params": sum(p.numel() for p in model.parameters()),
                               "accuracy": {k: _acc(v, rate) for k, v in acc.items()}}
    import shutil; shutil.rmtree(dep / "tmp", ignore_errors=True)
    json.dump(res, open(out / "deployment.json", "w"), indent=2)
    write_report(res, out)


def write_report(res: Dict, out: Path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"pytorch": "PyTorch (eager)", "torchscript": "TorchScript", "onnx": "ONNX Runtime fp32",
              "onnx_int8": "ONNX Runtime int8 (dynamic)"}
    L = ["# Deployment benchmark (design proposal, not a clinical claim)", "",
         f"CPU, 1 thread, batch size 1, one streaming step per frame; frame budget {res['budget_ms']:.1f} ms "
         "(32 Hz). Exported graph = one step (new frame + model state in, freeze logits + new state out). "
         "Outputs of the exported fp32 models match the offline PyTorch pass (max abs difference below). "
         "Accuracy change is measured on all 8 Daphnet test subjects (seed 0) with the decision threshold "
         "chosen on validation subjects using the fp32 ONNX model and then applied unchanged to int8.", ""]
    for name, r in res["models"].items():
        L += [f"## {name} ({r['params']:,} parameters)", "",
              "| Format | Median ms | p95 ms | Size (MB) | Max abs diff vs offline | Event F1 | Event recall | PR-AUC |",
              "|---|---|---|---|---|---|---|---|"]
        for k in labels:
            a = r["accuracy"].get(k)
            L.append(f"| {labels[k]} | {r['latency'][k]['median_ms']:.3f} | {r['latency'][k]['p95_ms']:.3f} | "
                     f"{r['size_mb'][k]:.3f} | {r['max_abs_diff_vs_offline'][k]:.1e} | "
                     + (f"{a['event_f1']:.1f} | {a['recall']:.1f} | {a['pr_auc']:.3f} |" if a else "n/a | n/a | n/a |"))
        if "onnx" in r["accuracy"]:
            d = r["accuracy"]["onnx_int8"]["event_f1"] - r["accuracy"]["onnx"]["event_f1"]
            L += ["", f"Event F1 change after int8 quantisation: {d:+.1f} points."]
        L.append("")
    (out / "deployment.md").write_text("\n".join(L) + "\n")
    fig, ax = plt.subplots(figsize=(7, 3.8)); names = list(res["models"]); w = 0.2
    for i, k in enumerate(labels):
        ax.bar(np.arange(len(names)) + (i - 1.5) * w, [res["models"][n]["latency"][k]["median_ms"] for n in names],
               w, label=labels[k])
    ax.axhline(res["budget_ms"], color="k", ls=":", label=f"frame budget {res['budget_ms']:.0f} ms")
    ax.set_yscale("log"); ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylabel("Median per-frame latency (ms)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Streaming latency by runtime (CPU, 1 thread)"); fig.tight_layout()
    fig.savefig(out / "deployment_latency.png", dpi=160)
    print(f"[done] {out / 'deployment.md'}, {out / 'deployment_latency.png'}")


if __name__ == "__main__":
    main()
