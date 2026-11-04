"""Calibration, fault-robustness and abstention evaluation on Daphnet (``main.py --robustness``).

Loads the models trained by the main benchmark (``results/daphnet/models``); nothing is retrained.
Everything that is a choice (temperature, decision threshold, abstention threshold, Freeze Index
thresholds) is made on the validation subjects of that fold, on clean data."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .calibration import calibration_report, fit_temperature, softmax
from .daphnet import SAMPLING_HZ, load_daphnet, loso_splits
from .data import Normalizer
from .fog import (event_counts, fi_scores, predict_logits, predict_logits_online_acausal,
                  select_fi_thresholds, select_threshold, summarize_counts)
from .fog_run import FI_NAME
from .models import build_model
from .robustness import CONDITIONS, apply_fault, cond_name

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]
MIN_COVERAGE = 0.8


def _logits(model, xs, args, hist, stride):
    if model.causal:
        return predict_logits(model, xs)
    return predict_logits_online_acausal(model, xs, hist, stride)


def _decide(p1: np.ndarray, thr: float, tau: float):
    """Alarm if p(freeze) >= thr AND the model is confident (max prob >= tau); else 'uncertain'
    frames raise no alarm. Returns (alarm, covered)."""
    conf = np.maximum(p1, 1 - p1)
    covered = conf >= tau
    return (p1 >= thr) & covered, covered


def _eval(p1s, ys, thr, tau, rate):
    alarms, covs = zip(*[_decide(p, thr, tau) for p in p1s])
    c = event_counts(list(alarms), ys, rate)
    cov = np.concatenate(covs)
    y, a = np.concatenate(ys).astype(bool), np.concatenate(alarms)
    c["covered"] = int(cov.sum()); c["total"] = int(len(cov))
    c["cov_correct"] = int(((a == y) & cov).sum())
    return c


def _run_seed(args, seed: int, threads: int):
    torch.set_num_threads(threads)
    out = Path(args.out)
    data = load_daphnet(args.data_dir, downsample=args.downsample)
    rate = SAMPLING_HZ / args.downsample
    hist, stride = int(round(args.online_history_s * rate)), args.recompute_stride
    cal_rows, abst_rows, fault_rows = [], [], []
    for fi_, (user, train, val, test) in enumerate(list(loso_splits(data, args.n_val, seed))[: args.max_folds]):
        t0 = time.time()
        val_y = [t.gestures for t in val]; te_y = [t.gestures for t in test]
        # ---- faulted copies of the raw test signals (identical for all models) -------------
        ck0 = torch.load(out / "models" / f"{args.models[0]}_seed{seed}_{user}.pt", weights_only=False)
        norm = Normalizer(mean=ck0["norm_mean"], std=ck0["norm_std"])
        faulted = {}
        for ci, (kind, level) in enumerate(CONDITIONS):
            faulted[(kind, level)] = [apply_fault(t.kinematics, kind, level, norm.std, rate,
                                                  np.random.default_rng([seed, fi_, ci, ti]))
                                      for ti, t in enumerate(test)]
        # ---- Freeze Index under faults ---------------------------------------------------
        th, pt = select_fi_thresholds(val, rate)
        for (kind, level), raws in faulted.items():
            from .daphnet import ANKLE_VERTICAL, freeze_index
            fp = [freeze_index(r[:, ANKLE_VERTICAL], rate) for r in raws]
            c = event_counts([(f > th) & (p > pt) for f, p in fp], te_y, rate)
            fault_rows.append({"seed": seed, "fold": user, "model": FI_NAME, "kind": kind, "level": level,
                               "mode": "plain", "counts": c})
        # ---- learned models --------------------------------------------------------------
        for m in args.models:
            ck = torch.load(out / "models" / f"{m}_seed{seed}_{user}.pt", weights_only=False)
            model = build_model(m, ck["cfg"]["n_features"], 2, ck["cfg"]["hidden"], ck["cfg"]["tcn_layers"])
            model.load_state_dict(ck["state"]); model.eval()
            val_x = [norm(t.kinematics) for t in val]
            lv = _logits(model, val_x, args, hist, stride)
            yv = np.concatenate(val_y)
            T = fit_temperature(np.concatenate(lv), yv)                      # validation only
            pv = [softmax(l, T)[:, 1] for l in lv]
            thr = select_threshold(pv, val_y, rate)                          # on calibrated val probs
            # abstention threshold: best val event F1 subject to coverage >= MIN_COVERAGE
            best, tau_star = -1.0, TAUS[0]
            for tau in TAUS:
                c = _eval(pv, val_y, thr, tau, rate)
                s = summarize_counts(c, rate)
                f1 = 0.0 if s["event_f1"] != s["event_f1"] else s["event_f1"]
                if c["covered"] / c["total"] >= MIN_COVERAGE and f1 > best + 1e-9:
                    best, tau_star = f1, tau
            base = {"seed": seed, "fold": user, "model": m}
            clean_logits = None
            for (kind, level), raws in faulted.items():
                lt = _logits(model, [norm(r) for r in raws], args, hist, stride)
                pt_ = [softmax(l, T)[:, 1] for l in lt]
                if kind == "clean":
                    clean_logits = np.concatenate(lt)
                    cal_rows.append({**base, "T": T, "thr": thr, "tau_star": tau_star,
                                     **calibration_report(clean_logits, np.concatenate(te_y), T)})
                    for tau in TAUS:
                        abst_rows.append({**base, "tau": tau, "counts": _eval(pt_, te_y, thr, tau, rate)})
                fault_rows.append({**base, "kind": kind, "level": level, "mode": "plain",
                                   "counts": _eval(pt_, te_y, thr, 0.5, rate)})
                fault_rows.append({**base, "kind": kind, "level": level, "mode": "abstain", "tau": tau_star,
                                   "counts": _eval(pt_, te_y, thr, tau_star, rate)})
        print(f"[robustness seed {seed}] fold {user} done ({time.time() - t0:.0f}s)", flush=True)
    return cal_rows, abst_rows, fault_rows


def run_robustness(args) -> Dict:
    from concurrent.futures import ProcessPoolExecutor
    out = Path(args.out)
    workers = max(1, min(args.workers, len(args.seeds)))
    threads = max(1, (__import__("os").cpu_count() or 1) // workers)
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_run_seed, [args] * len(args.seeds), args.seeds, [threads] * len(args.seeds)))
    res = {"calibration": [r for p in parts for r in p[0]], "abstention": [r for p in parts for r in p[1]],
           "faults": [r for p in parts for r in p[2]],
           "meta": {"taus": TAUS, "min_coverage": MIN_COVERAGE, "seeds": list(args.seeds),
                    "rate_hz": SAMPLING_HZ / args.downsample, "models": list(args.models),
                    "conditions": [cond_name(k, l) for k, l in CONDITIONS]}}
    with open(out / "robustness.json", "w") as f:
        json.dump(res, f, default=float)
    return res
