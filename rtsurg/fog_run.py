"""Daphnet LOSO benchmark driver: trains every model per (seed, fold), selects thresholds on
validation subjects, evaluates offline and online, adds the Freeze Index baseline, profiles
latency and writes ``results.json``."""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .benchmark import profile_latency
from .daphnet import SAMPLING_HZ, load_daphnet, loso_splits
from .data import Normalizer
from .fog import (average_precision, event_counts, fi_scores, freeze_prob, predict_logits,
                  predict_logits_online_acausal, select_fi_thresholds, select_threshold,
                  summarize_counts, train_fog_model)
from .models import MODEL_NAMES, build_model

FI_NAME = "Freeze Index"


def _eval_probs(probs: List[np.ndarray], trues: List[np.ndarray], thr: float, rate: float) -> Dict:
    c = event_counts([p >= thr for p in probs], trues, rate)
    return {"thr": thr, "counts": c, "summary": summarize_counts(c, rate),
            "pr_auc": average_precision(np.concatenate(probs), np.concatenate(trues))}


def _run_seed(args, seed: int, threads: int) -> List[Dict]:
    torch.set_num_threads(threads)
    out = Path(args.out)
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample
    hist_frames, stride = int(round(args.online_history_s * rate)), args.recompute_stride
    rows: List[Dict] = []
    for seed in [seed]:
        for user, train, val, test in list(loso_splits(data, args.n_val, seed))[: args.max_folds]:
            norm = Normalizer.fit(train)
            val_x = [norm(t.kinematics) for t in val]; val_y = [t.gestures for t in val]
            te_x = [norm(t.kinematics) for t in test]; te_y = [t.gestures for t in test]
            val_users = sorted({t.user for t in val})
            base = {"seed": seed, "fold": user, "val_users": val_users,
                    "train_users": sorted({t.user for t in train}),
                    "test_freeze_frac": 100 * float(np.concatenate(te_y).mean())}
            # ---- Freeze Index baseline (non-learned) ---------------------------------
            th, pt = select_fi_thresholds(val, rate)
            fi, pw = fi_scores(test, rate)
            c = event_counts([(f > th) & (p > pt) for f, p in zip(fi, pw)], te_y, rate)
            rows.append({**base, "model": FI_NAME, "mode": "online", "thr": th, "power_thr": pt,
                         "counts": c, "summary": summarize_counts(c, rate),
                         "pr_auc": average_precision(np.concatenate(fi), np.concatenate(te_y))})
            line = f"[seed {seed}] test {user} val {val_users} | FI f1 {rows[-1]['summary']['event_f1']:5.1f}"
            # ---- learned models ------------------------------------------------------
            for m in args.models:
                t0 = time.time()
                torch.manual_seed(seed)
                model = build_model(m, data.n_features, 2, args.hidden, args.tcn_layers)
                _, best_ap = train_fog_model(model, train, val, norm, rate, args.epochs, seed,
                                             win=args.window, stride=args.window, lr=args.lr,
                                             eval_every=3)
                torch.save({"state": model.state_dict(), "norm_mean": norm.mean, "norm_std": norm.std,
                            "cfg": {"hidden": args.hidden, "tcn_layers": args.tcn_layers, "model": m,
                                    "n_features": data.n_features}},
                           out / "models" / f"{m}_seed{seed}_{user}.pt")
                p_val = [freeze_prob(l) for l in predict_logits(model, val_x)]
                p_te = [freeze_prob(l) for l in predict_logits(model, te_x)]
                thr = select_threshold(p_val, val_y, rate)
                off = _eval_probs(p_te, te_y, thr, rate)
                rows.append({**base, "model": m, "mode": "offline", "val_pr_auc": best_ap, **off})
                if model.causal:
                    on = dict(off)                      # identical to streaming (see tests)
                else:
                    pv = [freeze_prob(l) for l in predict_logits_online_acausal(model, val_x, hist_frames, stride)]
                    pt_ = [freeze_prob(l) for l in predict_logits_online_acausal(model, te_x, hist_frames, stride)]
                    on = _eval_probs(pt_, te_y, select_threshold(pv, val_y, rate), rate)
                rows.append({**base, "model": m, "mode": "online", "val_pr_auc": best_ap, **on})
                line += (f" | {m} off {off['summary']['event_f1']:5.1f} on {on['summary']['event_f1']:5.1f}"
                         f" ({time.time() - t0:.0f}s)")
            print(line, flush=True)
    return rows


def run_daphnet(args) -> Dict:
    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample
    stride = args.recompute_stride
    print(f"[data] Daphnet: {len(data.trials)} segments, subjects {data.users}, "
          f"{sum(len(t.gestures) for t in data.trials)} frames at {rate:.0f} Hz "
          f"(frame budget {1000 / rate:.1f} ms)", flush=True)
    t_all = time.time()
    workers = max(1, min(args.workers, len(args.seeds)))
    threads = max(1, (os.cpu_count() or 1) // workers)
    if workers == 1:
        rows = [r for sd in args.seeds for r in _run_seed(args, sd, threads)]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for part in ex.map(_run_seed, [args] * len(args.seeds), args.seeds,
                                         [threads] * len(args.seeds)) for r in part]
    # ---- latency -----------------------------------------------------------------------
    hist_s = [10, 30, 60, 120, 300]
    lat = {}
    for m in args.models:
        model = build_model(m, data.n_features, 2, args.hidden, args.tcn_layers)
        lat[m] = profile_latency(model, data.n_features, "cpu", [int(s * rate) for s in hist_s])
    meta = {"dataset": "Daphnet Freezing of Gait", "rate_hz": rate, "frame_ms": 1000 / rate,
            "seeds": list(args.seeds), "epochs": args.epochs, "hidden": args.hidden,
            "tcn_layers": args.tcn_layers, "n_val": args.n_val, "window": args.window,
            "lr": args.lr, "online_history_s": args.online_history_s,
            "recompute_stride": stride, "merge_gap_s": 1.0, "downsample": args.downsample,
            "torch": torch.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "hist_s": hist_s,
            "runtime_s": time.time() - t_all}
    res = {"meta": meta, "rows": rows, "latency": lat, "models": list(args.models)}
    with open(out / "results.json", "w") as f:
        json.dump(res, f, default=float)
    return res
