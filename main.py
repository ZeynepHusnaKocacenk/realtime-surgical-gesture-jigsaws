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
                         generate_synthetic_jigsaws, load_jigsaws_task, loso_supertrial_splits, louo_splits)
from rtsurg.models import MODEL_NAMES, build_model
from rtsurg.train import TrainConfig, train_model

FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, _ = pre.parse_known_args()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["jigsaws", "daphnet"], default="jigsaws",
                   help="daphnet: --data-dir points at the folder holding SxxRyy.txt files")
    p.add_argument("--config", type=str, default=None, help="YAML file of settings (see configs/)")
    p.add_argument("--skip-if-reproduced", action="store_true",
                   help="if saved results already come from this config, only rebuild tables and figures")
    src = p.add_mutually_exclusive_group(required=False)
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
    p.add_argument("--protocol", choices=["louo", "loso"], default="louo",
                   help="JIGSAWS: louo = leave one surgeon out; loso = leave one supertrial (repetition) out")
    p.add_argument("--max-folds", type=int, default=None, help="limit LOUO folds (quick runs)")
    p.add_argument("--latency-device", type=str, default="cpu",
                   help="device for latency profiling (cpu mimics an embedded OR workstation)")
    p.add_argument("--out", type=str, default="results")
    d = p.add_argument_group("Daphnet options")
    d.add_argument("--robustness", action="store_true",
                   help="calibration, fault robustness and abstention on the saved Daphnet models")
    d.add_argument("--workers", type=int, default=3, help="parallel processes (one per seed)")
    d.add_argument("--n-val", type=int, default=2, help="validation subjects per fold")
    d.add_argument("--window", type=int, default=256, help="training window (frames)")
    d.add_argument("--online-history-s", type=float, default=10.0,
                   help="history an acausal model may re-process when run online")
    d.add_argument("--daphnet-hidden", type=int, default=32)
    d.add_argument("--daphnet-tcn-layers", type=int, default=6)
    d.add_argument("--daphnet-epochs", type=int, default=12)
    d.add_argument("--daphnet-lr", type=float, default=2e-3)
    d.add_argument("--daphnet-downsample", type=int, default=2, help="64 Hz / k")
    d.add_argument("--features", choices=["raw", "spectral"], default="raw",
                   help="model input: raw accelerations, or raw + causal band powers (rtsurg/features.py)")
    d.add_argument("--augment-noise", type=float, default=0.0,
                   help="add one noisy copy of every training trial, noise std U(0.1, x) x feature std (0 = off)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="random seeds; results are mean and std across seeds and folds")
    p.add_argument("--verbose", action="store_true")
    if known.config:
        import yaml
        cfg = yaml.safe_load(open(known.config))
        unknown = [k for k in cfg if not any(a.dest == k for a in p._actions)]
        if unknown:
            raise SystemExit(f"{known.config}: unknown setting(s) {unknown}")
        p.set_defaults(**cfg)
    args = p.parse_args()
    if not (args.data_dir or args.synthetic):
        p.error("one of --data-dir / --synthetic (or a config with data_dir) is required")
    return args


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a))}


def main():
    args = parse_args()
    if args.dataset == "daphnet":
        from rtsurg.fog_report import write_daphnet_report
        from rtsurg.fog_run import run_daphnet
        if args.synthetic or not args.data_dir:
            raise SystemExit("--dataset daphnet needs --data-dir (real Daphnet files); no synthetic fallback")
        if args.out == "results":
            args.out = "results/daphnet"
        args.epochs, args.lr, args.downsample = args.daphnet_epochs, args.daphnet_lr, args.daphnet_downsample
        args.hidden, args.tcn_layers = args.daphnet_hidden, args.daphnet_tcn_layers
        args.recompute_stride = args.recompute_stride if args.recompute_stride != 5 else 8
        from rtsurg.config import daphnet_results_match, robustness_results_match
        if args.robustness:
            from rtsurg.robust_report import write_robustness_report
            from rtsurg.robust_run import run_robustness
            saved = Path(args.out) / "robustness.json"
            if args.skip_if_reproduced and robustness_results_match(args, saved):
                print(f"[reproduce] {saved} already comes from this config; rebuilding tables only")
                write_robustness_report(json.load(open(saved)), Path(args.out))
                return
            if not (Path(args.out) / "models").exists():
                raise SystemExit("no trained models found: run the Daphnet benchmark first")
            write_robustness_report(run_robustness(args), Path(args.out))
            return
        saved = Path(args.out) / "results.json"
        if args.skip_if_reproduced and daphnet_results_match(args, saved):
            print(f"[reproduce] {saved} already comes from this config (same seeds and settings); "
                  "not re-training, rebuilding tables and figures only")
            write_daphnet_report(json.load(open(saved)), Path(args.out))
            return
        res = run_daphnet(args)
        write_daphnet_report(res, Path(args.out))
        return
    out = Path(args.out)
    if not args.synthetic and not Path(args.data_dir).exists():
        out.mkdir(parents=True, exist_ok=True)
        (out / "AWAITING_DATASET").write_text(f"JIGSAWS folder {args.data_dir} not found: awaiting dataset access\n")
        print(f"[jigsaws] {args.data_dir} not found: awaiting dataset access, nothing was run")
        return
    (out / "AWAITING_DATASET").unlink(missing_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds)
    torch.manual_seed(seeds[0]); np.random.seed(seeds[0])

    if args.synthetic:
        task_dir = generate_synthetic_jigsaws(out / "synthetic_data", seed=seeds[0])
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

    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=seeds[0])
    splitter = louo_splits if args.protocol == "louo" else loso_supertrial_splits
    folds = list(splitter(data))[: args.max_folds]
    per_model = {m: {"folds": [], "history": []} for m in args.models}

    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed); cfg.seed = seed
        for user, train_trials, test_trials in folds:
            norm = Normalizer.fit(train_trials)                     # train-fold statistics only
            train_ds = TrialDataset(train_trials, norm, args.downsample)
            test_ds = TrialDataset(test_trials, norm, args.downsample)
            train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_padded)
            test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_padded)
            print(f"\n[seed {seed}] [fold] held out {user} ({skills.get(user, args.protocol.upper())}): "
                  f"{len(train_trials)} train / {len(test_trials)} test trials")
            for m in args.models:
                t0 = time.time()
                model = build_model(m, data.n_features, data.n_gestures, args.hidden, args.tcn_layers)
                hist = train_model(model, train_dl, test_dl, cfg, verbose=args.verbose)
                res = evaluate_fold(model, test_ds, cfg.device, frame_ms, args.recompute_stride, FRACTIONS)
                res["user"], res["seed"] = user, seed
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
            "folds": [u for u, _, _ in folds], "seeds": seeds, "epochs": args.epochs, "torch": torch.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "recompute_stride": args.recompute_stride, "manipulators": args.manipulators,
            "protocol": args.protocol}
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
             "", f"{meta.get('protocol', 'louo').upper()} folds: {', '.join(meta['folds'])} · {meta['rate_hz']:.0f} Hz · "
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
