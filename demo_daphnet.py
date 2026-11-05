"""Real-data streaming demo: replay one held-out Daphnet patient frame by frame through a trained
causal TCN, as a wearable gateway would, then write a timeline figure, a session file and a
guard-railed session summary (``rtsurg/report_agent.py``).

    python demo_daphnet.py --data-dir data/daphnet                 # default: patient S05, shipped model
    python demo_daphnet.py --data-dir data/daphnet --subject S02 --checkpoint results/daphnet_spectral/models/TCN-causal_seed0_S02.pt

The shipped checkpoint (``assets/``) was trained in the leave-one-subject-out benchmark WITHOUT the
demo patient; its threshold and debounce were chosen on that fold's validation patients. The
Daphnet data itself is not in the repository (download it from the UCI repository).
Research prototype only: no clinical claims.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rtsurg.daphnet import SAMPLING_HZ, load_daphnet
from rtsurg.data import Normalizer
from rtsurg.features import featurize
from rtsurg.fog import freeze_prob, postprocess, predict_logits
from rtsurg.models import build_model

ROOT = Path(__file__).resolve().parent


def load_operating_point(ckpt: Path, explicit: str | None = None) -> dict:
    """Threshold + debounce chosen on that fold's VALIDATION patients: from a sidecar JSON, or
    looked up in the ``results.json`` of the run that produced ``<run>/models/<model>_seed<s>_<fold>.pt``."""
    side = Path(explicit) if explicit else ckpt.with_suffix(".json")
    if side.exists():
        return json.load(open(side))
    model, rest = ckpt.stem.rsplit("_seed", 1)
    seed, fold = rest.split("_")
    res = json.load(open(ckpt.parent.parent / "results.json"))
    row = next(r for r in res["rows"] if r["model"] == model and r["mode"] == "online_pp"
               and r["seed"] == int(seed) and r["fold"] == fold)
    return {"thr": row["thr"], "min_dur": row["min_dur"], "fold": fold, "seed": int(seed), "model_name": model,
            "val_users": row["val_users"]}


def stream_patient(ck, trials, rate):
    cfg = ck["cfg"]
    model = build_model(cfg["model"], cfg["n_features"], 2, cfg["hidden"], cfg["tcn_layers"])
    model.load_state_dict(ck["state"]); model.eval()
    norm = Normalizer(mean=np.asarray(ck["norm_mean"]), std=np.asarray(ck["norm_std"]))
    probs, lat, max_diff = [], [], 0.0
    for t in trials:                                   # each recording segment starts a fresh stream
        x = torch.from_numpy(norm(featurize(t.kinematics, cfg.get("features", "raw"), rate)))
        model.stream_reset()
        p = np.empty(len(x))
        for i in range(len(x)):
            t0 = time.perf_counter()
            g, _ = model.stream_step(x[i])
            lat.append((time.perf_counter() - t0) * 1000.0)
            p[i] = torch.softmax(g, -1)[1].item()
        off = freeze_prob(predict_logits(model, [x.numpy()])[0])
        max_diff = max(max_diff, float(np.abs(off - p).max()))
        probs.append(p)
    return probs, np.asarray(lat), max_diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "daphnet"))
    ap.add_argument("--subject", default="S05")
    ap.add_argument("--checkpoint", default=str(ROOT / "assets" / "tcn_spectral_seed0_S05.pt"))
    ap.add_argument("--operating-point", default=None,
                    help="JSON with thr and min_dur (default: <checkpoint>.json, or results.json of the run)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "demo"))
    ap.add_argument("--offline-writer", action="store_true", help="use the template writer even if an API key is set")
    args = ap.parse_args(argv)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(args.checkpoint, weights_only=False)
    op = load_operating_point(Path(args.checkpoint), args.operating_point)
    data = load_daphnet(args.data_dir, downsample=2)
    rate = SAMPLING_HZ / 2
    trials = [t for t in data.trials if t.user == args.subject]
    if not trials:
        raise SystemExit(f"no recordings for {args.subject}")
    if op.get("fold") not in (None, args.subject):
        print(f"[warning] the checkpoint was trained with {args.subject} in its TRAINING set "
              f"(it was held out for {op['fold']}); results on {args.subject} are optimistic")
    print(f"streaming {args.subject}: {len(trials)} segments, {sum(len(t.gestures) for t in trials)} frames at "
          f"{rate:.0f} Hz, threshold {op['thr']}, debounce {op['min_dur'] / rate:.2f} s")
    probs, lat, max_diff = stream_patient(ck, trials, rate)
    alarms = [postprocess(p >= op["thr"], op["min_dur"]) for p in probs]
    budget = 1000.0 / rate
    print(f"per-frame latency median {np.median(lat):.3f} ms, p99 {np.percentile(lat, 99):.3f} ms, "
          f"{100 * np.mean(lat <= budget):.2f}% of {len(lat)} frames within the {budget:.2f} ms budget; "
          f"streaming vs offline max |diff| {max_diff:.1e}")

    # ---- session file + summary ---------------------------------------------------------
    cat = lambda xs: np.concatenate(xs)
    session = {"rate_hz": rate, "prob": cat(probs), "alarm": cat(alarms), "labels": cat([t.gestures for t in trials]),
               "raw": cat([t.kinematics for t in trials]), "subject": args.subject,
               "model": op.get("model_name", ck["cfg"]["model"]), "threshold": op["thr"],
               "debounce_s": op["min_dur"] / rate, "latency_ms": float(np.median(lat))}
    from rtsurg.report_agent import default_llm, write_report
    rep = write_report(session, default_llm(offline=args.offline_writer))
    (out / f"session_{args.subject}.md").write_text(rep["report"] + "\n")
    json.dump({"facts": rep["facts"], "trace": rep["trace"], "source": rep["source"],
               "latency_ms": {"median": float(np.median(lat)), "p99": float(np.percentile(lat, 99)),
                              "max": float(lat.max()), "budget": budget},
               "stream_vs_offline_max_abs_diff": max_diff}, open(out / f"session_{args.subject}.json", "w"), indent=1)
    print(f"\n{rep['report']}\n\n[agent trace] {' -> '.join(rep['trace'])}")

    # ---- figure: the segment with the most freezing --------------------------------------
    k = int(np.argmax([t.gestures.sum() for t in trials]))
    t, p, a = trials[k], probs[k], alarms[k]
    s = max(int(np.flatnonzero(t.gestures)[0] - 30 * rate), 0)
    e = min(s + int(180 * rate), len(p))
    tt = np.arange(s, e) / rate
    fig, ax = plt.subplots(3, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 0.5]})
    ax[0].plot(tt, t.kinematics[s:e, 1], lw=0.5, color="#555")
    ax[0].set_ylabel("ankle vertical\nacc. (raw)")
    ax[1].plot(tt, p[s:e], lw=0.9, color="#1f77b4", label="p(freeze), streamed")
    ax[1].axhline(op["thr"], color="#1f77b4", ls=":", lw=0.8, label=f"threshold {op['thr']}")
    ax[1].set_ylim(0, 1); ax[1].set_ylabel("probability"); ax[1].legend(loc="upper right", fontsize=7, frameon=False)
    ax[2].fill_between(tt, 0.55, 0.95, where=t.gestures[s:e] == 1, step="post", color="#d62728", label="annotated freeze")
    ax[2].fill_between(tt, 0.05, 0.45, where=a[s:e], step="post", color="#1f77b4", label="alarm (debounced)")
    ax[2].set_yticks([]); ax[2].set_xlabel("time in recording segment (s)")
    ax[2].legend(loc="lower right", fontsize=7, ncol=1, frameon=True, framealpha=0.9)
    for i in (0, 1):
        ax[i].fill_between(tt, *ax[i].get_ylim(), where=t.gestures[s:e] == 1, step="post", color="#d62728", alpha=0.08)
    fig.suptitle(f"Daphnet patient {args.subject} (held out), streamed frame by frame: median "
                 f"{np.median(lat):.2f} ms per frame vs a {budget:.1f} ms budget", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"stream_{args.subject}.png", dpi=150); plt.close(fig)
    print(f"[done] {out / f'stream_{args.subject}.png'}, {out / f'session_{args.subject}.md'}")


if __name__ == "__main__":
    main()
