"""Run the full LOUO benchmark: load data -> train each model per fold -> offline & online
evaluation -> latency profiling -> results table + figures.

Examples
--------
Pipeline smoke test on synthetic JIGSAWS-format data (minutes on a laptop CPU):
    python main.py --synthetic --epochs 15 --max-folds 3

Real JIGSAWS (point at one task folder):
    python main.py --data-dir /path/to/JIGSAWS/Suturing --epochs 60
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from rtsurg import plots
from rtsurg.benchmark import evaluate_fold, profile_latency
from rtsurg.data import (FRAME_RATE_HZ, IDX_TO_SKILL, Normalizer, TrialDataset, collate_padded,
                         generate_synthetic_jigsaws, load_jigsaws_task, louo_splits)
from rtsurg.models import MODEL_NAMES, build_model
from rtsurg.train import TrainConfig, train_model

FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", type=str, help="JIGSAWS task folder, e.g. .../JIGSAWS/Suturing")
    src.add_argument("--synthetic", action="store_true", help="generate mock data in JIGSAWS format")
    p.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    p.add_argument("--manipulators", nargs="+", default=["PSM_L", "PSM_R"],
                   choices=["MTM_L", "MTM_R", "PSM_L", "PSM_R"])
    p.add_argument("--downsample", type=int, default=3, help="30 Hz / k (default 3 -> 10 Hz)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--tcn-layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recompute-stride", type=int, default=5,
                   help="online acausal models are re-run every k frames")
    p.add_argument("--max-folds", type=int, default=None, help="limit LOUO folds (quick runs)")
    p.add_argument("--latency-device", type=str, default="cpu",
                   help="device for latency profiling (cpu mimics an embedded OR workstation)")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a))}


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); (out / "figures").mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        task_dir = generate_synthetic_jigsaws(out / "synthetic_data", seed=args.seed)
        print(f"[data] synthetic JIGSAWS-format data written to {task_dir}")
    else:
        task_dir = Path(args.data_dir)
    data = load_jigsaws_task(task_dir, manipulators=args.manipulators)
    rate = FRAME_RATE_HZ / args.downsample
    frame_ms = 1000.0 / rate
    print(f"[data] task={data.trials[0].task} trials={len(data.trials)} surgeons={data.users} "
          f"features={data.n_features} gestures={data.n_gestures} rate={rate:.0f} Hz")
    skills = {u: IDX_TO_SKILL[next(t.skill for t in data.trials if t.user == u)] for u in data.users}
    print(f"[data] skill per surgeon: {skills}")

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed)
    folds = list(louo_splits(data))[: args.max_folds]
    per_model = {m: {"folds": [], "history": []} for m in args.models}

    for user, train_trials, test_trials in folds:
        norm = Normalizer.fit(train_trials)                     # train-fold statistics only
        train_ds = TrialDataset(train_trials, norm, args.downsample)
        test_ds = TrialDataset(test_trials, norm, args.downsample)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_padded)
        test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_padded)
        print(f"\n[fold] held-out surgeon {user} ({skills[user]}): "
              f"{len(train_trials)} train / {len(test_trials)} test trials")
        for m in args.models:
            t0 = time.time()
            model = build_model(m, data.n_features, data.n_gestures, args.hidden, args.tcn_layers)
            hist = train_model(model, train_dl, test_dl, cfg, verbose=args.verbose)
            res = evaluate_fold(model, test_ds, cfg.device, frame_ms, args.recompute_stride, FRACTIONS)
            res["user"] = user
            per_model[m]["folds"].append(res)
            per_model[m]["history"].append(hist.__dict__)
            print(f"  {m:12s} offline acc {res['offline']['frame_acc']:5.1f} | online acc "
                  f"{res['online']['frame_acc']:5.1f} edit {res['online']['edit']:5.1f} | "
                  f"skill {res['skill_acc']:5.1f} | {time.time() - t0:5.1f}s")

    # ---- latency (architecture-dependent, independent of the fold) ------------------------
    history_s = [10, 30, 60, 120, 300]
    history_frames = [int(s * rate) for s in history_s]
    ref_history = int(60 * rate)
    print(f"\n[latency] device={args.latency_device}, frame budget {frame_ms:.0f} ms at {rate:.0f} Hz")
    summary = {}
    for m in args.models:
        model = build_model(m, data.n_features, data.n_gestures, args.hidden, args.tcn_layers)
        lat = profile_latency(model, data.n_features, args.latency_device, history_frames)
        folds_r = per_model[m]["folds"]
        delays = [d for f in folds_r for d in f["delay_ms"]]
        summary[m] = {
            "causal": model.causal,
            "params": sum(p.numel() for p in model.parameters()),
            "offline": {k: mean_std([f["offline"][k] for f in folds_r]) for k in folds_r[0]["offline"]},
            "online": {k: mean_std([f["online"][k] for f in folds_r]) for k in folds_r[0]["online"]},
            "skill_acc": mean_std([f["skill_acc"] for f in folds_r]),
            "early_skill": {p: mean_std([f["early_skill"][p] for f in folds_r]) for p in FRACTIONS},
            "median_delay_ms": float(np.median(delays)) if delays else float("nan"),
            "detected_frac": float(np.mean([f["detected_frac"] for f in folds_r])),
            "latency": lat,
        }
        r = lat["by_history"][ref_history]
        print(f"  {m:12s} median {r['median_ms']:7.2f} ms  p95 {r['p95_ms']:7.2f} ms  (at 60 s history)")

    # ---- outputs -------------------------------------------------------------------------
    meta = {"task": data.trials[0].task, "synthetic": args.synthetic, "rate_hz": rate,
            "folds": [u for u, _, _ in folds], "epochs": args.epochs, "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "recompute_stride": args.recompute_stride, "manipulators": args.manipulators}
    with open(out / "results.json", "w") as f:
        json.dump({"meta": meta, "summary": summary,
                   "per_fold": {m: [{k: v for k, v in fr.items() if k not in ("example", "delay_ms")}
                                    for fr in per_model[m]["folds"]] for m in args.models},
                   "history": {m: per_model[m]["history"] for m in args.models}}, f, indent=2, default=str)
    write_markdown(summary, meta, ref_history, frame_ms, out / "results.md")

    fig = out / "figures"
    plots.accuracy_vs_latency(summary, frame_ms, ref_history, fig / "accuracy_vs_latency.png")
    plots.offline_vs_online(summary, fig / "offline_vs_online.png")
    plots.latency_vs_history(summary, frame_ms, frame_ms, fig / "latency_vs_history.png")
    plots.early_skill(summary, fig / "early_skill.png")
    plots.gesture_timeline({m: per_model[m]["folds"][0]["example"] for m in args.models},
                           data.gesture_names, frame_ms, fig / "gesture_timeline.png")
    print(f"\n[done] results -> {out/'results.md'}, figures -> {fig}")


def write_markdown(summary, meta, ref_history, frame_ms, path: Path):
    def ms(d):
        return f"{d['mean']:.1f} ± {d['std']:.1f}"
    lines = [f"# Results: {meta['task']} ({'SYNTHETIC DATA, pipeline test only' if meta['synthetic'] else 'JIGSAWS'})",
             "", f"LOUO folds: {', '.join(meta['folds'])} · {meta['rate_hz']:.0f} Hz · "
             f"frame budget {frame_ms:.0f} ms · latency on `{meta['processor'] or meta['platform']}`", "",
             "## Gesture recognition (mean ± std over folds)", "",
             "| Model | Causal | Offline acc | Online acc | Online edit | Online F1@10 | Online F1@25 | Online F1@50 |",
             "|---|---|---|---|---|---|---|---|"]
    for m, r in summary.items():
        o, n = r["offline"], r["online"]
        lines.append(f"| {m} | {'yes' if r['causal'] else 'no'} | {ms(o['frame_acc'])} | {ms(n['frame_acc'])} | "
                     f"{ms(n['edit'])} | {ms(n['f1@10'])} | {ms(n['f1@25'])} | {ms(n['f1@50'])} |")
    lines += ["", "## Real-time behaviour", "",
              "| Model | Params | Latency median / p95 (ms, 60 s history) | Gesture detection delay (median ms) | Gestures detected | Skill acc (full trial) | Skill acc (first 25%) |",
              "|---|---|---|---|---|---|---|"]
    for m, r in summary.items():
        lat = r["latency"]["by_history"][ref_history]
        lines.append(f"| {m} | {r['params']:,} | {lat['median_ms']:.2f} / {lat['p95_ms']:.2f} | "
                     f"{r['median_delay_ms']:.0f} | {r['detected_frac']*100:.0f}% | {ms(r['skill_acc'])} | "
                     f"{ms(r['early_skill'][0.25])} |")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
