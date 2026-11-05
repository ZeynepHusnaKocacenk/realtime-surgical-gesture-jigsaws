"""Daphnet LOSO benchmark driver: trains every model per (seed, fold), selects thresholds on
validation subjects, evaluates offline and online, adds the Freeze Index baseline, profiles
latency and writes ``results.json``.

Rows per (seed, fold, model):
  mode ``offline`` / ``online``  threshold only (the original protocol);
  mode ``online_pp``             threshold + debounce duration, both chosen on validation subjects
                                 (``rtsurg.fog.postprocess``), for the learned models AND the Freeze Index.
Every row also stores ``counts_strict`` (one-to-one event matching, a sensitivity check).
"""
from __future__ import annotations

import dataclasses
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
from .features import featurize
from .fog import (MIN_DUR_S, average_precision, event_counts, fi_gated_score, fi_scores, freeze_prob,
                  postprocess, predict_logits, predict_logits_online_acausal, select_fi_thresholds,
                  select_operating_point, select_threshold, summarize_counts, train_fog_model)
from .models import MODEL_NAMES, build_model

FI_NAME = "Freeze Index"


def _counts_both(preds, trues, rate):
    return event_counts(preds, trues, rate), event_counts(preds, trues, rate, strict=True)


def _eval_probs(probs: List[np.ndarray], trues: List[np.ndarray], thr: float, rate: float,
                min_dur: int = 0) -> Dict:
    c, cs = _counts_both([postprocess(p >= thr, min_dur) for p in probs], trues, rate)
    return {"thr": thr, "min_dur": min_dur, "counts": c, "counts_strict": cs,
            "summary": summarize_counts(c, rate),
            "pr_auc": average_precision(np.concatenate(probs), np.concatenate(trues))}


def _augment(trials, norm_std, sigma_max: float, rng):
    """One noisy copy of every training trial: Gaussian noise with std U(0.1, sigma_max) x the
    per-feature training std, added to the RAW signal (the fault model of the robustness study)."""
    out = []
    for t in trials:
        s = rng.uniform(0.1, sigma_max)
        x = t.kinematics + rng.normal(size=t.kinematics.shape).astype(np.float32) * (s * norm_std)
        out.append(dataclasses.replace(t, name=t.name + "_aug", kinematics=x.astype(np.float32)))
    return out


def _run_seed(args, seed: int, threads: int) -> List[Dict]:
    torch.set_num_threads(threads)
    out = Path(args.out)
    feats = getattr(args, "features", "raw")
    aug = float(getattr(args, "augment_noise", 0.0) or 0.0)
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample
    hist_frames, stride = int(round(args.online_history_s * rate)), args.recompute_stride
    fz = lambda ts: [dataclasses.replace(t, kinematics=featurize(t.kinematics, feats, rate)) for t in ts]
    rows: List[Dict] = []
    for user, train, val, test in list(loso_splits(data, args.n_val, seed))[: args.max_folds]:
        raw_std = Normalizer.fit(train).std
        train_f, val_f, test_f = fz(train), fz(val), fz(test)
        if aug > 0:
            train_f = train_f + fz(_augment(train, raw_std, aug, np.random.default_rng([seed, 7, len(rows)])))
        norm = Normalizer.fit(fz(train))                    # statistics of the clean training subjects
        val_x = [norm(t.kinematics) for t in val_f]; val_y = [t.gestures for t in val]
        te_x = [norm(t.kinematics) for t in test_f]; te_y = [t.gestures for t in test]
        val_users = sorted({t.user for t in val})
        base = {"seed": seed, "fold": user, "val_users": val_users,
                "train_users": sorted({t.user for t in train}),
                "test_freeze_frac": 100 * float(np.concatenate(te_y).mean())}
        # ---- Freeze Index baseline (non-learned) -------------------------------------
        th, pt = select_fi_thresholds(val, rate)
        fi, pw = fi_scores(test, rate)
        gated = [fi_gated_score(f, p, pt) for f, p in zip(fi, pw)]
        raw_alarm = [(f > th) & (p > pt) for f, p in zip(fi, pw)]
        c, cs = _counts_both(raw_alarm, te_y, rate)
        fi_auc = average_precision(np.concatenate(gated), np.concatenate(te_y))
        fi_auc_ungated = average_precision(np.concatenate(fi), np.concatenate(te_y))
        rows.append({**base, "model": FI_NAME, "mode": "online", "thr": th, "power_thr": pt, "min_dur": 0,
                     "counts": c, "counts_strict": cs, "summary": summarize_counts(c, rate),
                     "pr_auc": fi_auc, "pr_auc_ungated": fi_auc_ungated})
        th2, pt2, md = select_fi_thresholds(val, rate, MIN_DUR_S)
        alarm2 = [postprocess((f > th2) & (p > pt2), md) for f, p in zip(fi, pw)]
        c2, cs2 = _counts_both(alarm2, te_y, rate)
        rows.append({**base, "model": FI_NAME, "mode": "online_pp", "thr": th2, "power_thr": pt2, "min_dur": md,
                     "counts": c2, "counts_strict": cs2, "summary": summarize_counts(c2, rate),
                     "pr_auc": average_precision(np.concatenate([fi_gated_score(f, p, pt2) for f, p in zip(fi, pw)]),
                                                 np.concatenate(te_y))})
        line = (f"[seed {seed}] test {user} val {val_users} | FI f1 {rows[-2]['summary']['event_f1']:5.1f}"
                f" pp {rows[-1]['summary']['event_f1']:5.1f}")
        # ---- learned models ----------------------------------------------------------
        for m in args.models:
            t0 = time.time()
            torch.manual_seed(seed)
            n_feat = train_f[0].kinematics.shape[1]
            model = build_model(m, n_feat, 2, args.hidden, args.tcn_layers)
            _, best_ap = train_fog_model(model, train_f, val_f, norm, rate, args.epochs, seed,
                                         win=args.window, stride=args.window, lr=args.lr, eval_every=3)
            torch.save({"state": model.state_dict(), "norm_mean": norm.mean, "norm_std": norm.std,
                        "cfg": {"hidden": args.hidden, "tcn_layers": args.tcn_layers, "model": m,
                                "n_features": n_feat, "features": feats, "raw_std": raw_std}},
                       out / "models" / f"{m}_seed{seed}_{user}.pt")
            if model.causal:
                p_val = [freeze_prob(l) for l in predict_logits(model, val_x)]
                p_te = [freeze_prob(l) for l in predict_logits(model, te_x)]
                pv_on, pt_on = p_val, p_te                    # identical to streaming (see tests)
            else:
                p_val = [freeze_prob(l) for l in predict_logits(model, val_x)]
                p_te = [freeze_prob(l) for l in predict_logits(model, te_x)]
                pv_on = [freeze_prob(l) for l in predict_logits_online_acausal(model, val_x, hist_frames, stride)]
                pt_on = [freeze_prob(l) for l in predict_logits_online_acausal(model, te_x, hist_frames, stride)]
            off = _eval_probs(p_te, te_y, select_threshold(p_val, val_y, rate), rate)
            on = _eval_probs(pt_on, te_y, select_threshold(pv_on, val_y, rate), rate)
            thr_pp, md = select_operating_point(pv_on, val_y, rate)
            pp = _eval_probs(pt_on, te_y, thr_pp, rate, md)
            for mode, r in (("offline", off), ("online", on), ("online_pp", pp)):
                rows.append({**base, "model": m, "mode": mode, "val_pr_auc": best_ap, **r})
            line += (f" | {m} off {off['summary']['event_f1']:5.1f} on {on['summary']['event_f1']:5.1f}"
                     f" pp {pp['summary']['event_f1']:5.1f} ({time.time() - t0:.0f}s)")
        print(line, flush=True)
    return rows


def run_daphnet(args) -> Dict:
    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample
    stride = args.recompute_stride
    feats = getattr(args, "features", "raw")
    print(f"[data] Daphnet: {len(data.trials)} segments, subjects {data.users}, "
          f"{sum(len(t.gestures) for t in data.trials)} frames at {rate:.0f} Hz "
          f"(frame budget {1000 / rate:.1f} ms), features {feats}", flush=True)
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
    n_feat = featurize(data.trials[0].kinematics[:200], feats, rate).shape[1]
    lat = {}
    for m in args.models:
        model = build_model(m, n_feat, 2, args.hidden, args.tcn_layers)
        lat[m] = profile_latency(model, n_feat, "cpu", [int(s * rate) for s in hist_s])
    meta = {"dataset": "Daphnet Freezing of Gait", "rate_hz": rate, "frame_ms": 1000 / rate,
            "seeds": list(args.seeds), "epochs": args.epochs, "hidden": args.hidden,
            "tcn_layers": args.tcn_layers, "n_val": args.n_val, "window": args.window,
            "lr": args.lr, "online_history_s": args.online_history_s,
            "recompute_stride": stride, "merge_gap_s": 1.0, "downsample": args.downsample,
            "features": feats, "augment_noise": float(getattr(args, "augment_noise", 0.0) or 0.0),
            "min_dur_grid_s": list(MIN_DUR_S),
            "torch": torch.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "hist_s": hist_s,
            "runtime_s": time.time() - t_all}
    res = {"meta": meta, "rows": rows, "latency": lat, "models": list(args.models)}
    with open(out / "results.json", "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    return res
