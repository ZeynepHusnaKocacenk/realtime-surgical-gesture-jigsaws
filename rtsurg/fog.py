"""Freezing-of-gait benchmark on Daphnet: training, validation-only threshold selection,
event-level metrics, offline vs. online evaluation and the Freeze Index baseline.

Protocol (per seed, per fold):
  test subject  = the held-out subject (never used for any choice);
  validation    = ``n_val`` other subjects with freezes: used for epoch selection and for the
                  decision threshold (and the Freeze Index thresholds);
  training      = all remaining subjects.

Event definitions (documented in the README):
  * true episode      = a maximal run of freeze frames;
  * predicted event   = a maximal run of frames with p(freeze) >= threshold, where runs separated
                        by <= ``MERGE_GAP_S`` seconds are merged;
  * episode detected  = any predicted event overlaps it (recall = detected / episodes);
  * predicted event is correct if at least half of its frames are freeze frames (precision =
                        correct / predicted); this stops an always-on detector from scoring well;
  * false alarm       = a predicted event that is not correct;
  * detection delay   = time from episode onset to the first positive frame inside it.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .daphnet import ANKLE_VERTICAL, _runs, freeze_index
from .data import Normalizer, Trial
from .models import SeqModel

MERGE_GAP_S = 1.0
MIN_EVENT_PRECISION = 0.5   # a predicted event is correct only if >= 50% of its frames are freeze
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


# ---------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------
def pred_events(pred: np.ndarray, merge_gap: int) -> List[Tuple[int, int]]:
    ev: List[List[int]] = []
    for s, e in _runs(pred.astype(bool)):
        if ev and s - ev[-1][1] <= merge_gap:
            ev[-1][1] = e
        else:
            ev.append([s, e])
    return [tuple(v) for v in ev]


def event_counts(preds: Sequence[np.ndarray], trues: Sequence[np.ndarray], rate_hz: float) -> Dict:
    """Raw counts so that results can be pooled over trials, subjects and folds."""
    gap = int(round(MERGE_GAP_S * rate_hz))
    c = dict(n_true=0, detected=0, n_pred=0, correct=0, false_alarms=0, frames=0,
             tp=0, fn=0, fp=0, tn=0, delays_ms=[])
    for p, t in zip(preds, trues):
        p, t = p.astype(bool), t.astype(bool)
        truth = _runs(t)
        evs = pred_events(p, gap)
        c["frames"] += len(t)
        c["tp"] += int((p & t).sum()); c["fn"] += int((~p & t).sum())
        c["fp"] += int((p & ~t).sum()); c["tn"] += int((~p & ~t).sum())
        c["n_true"] += len(truth); c["n_pred"] += len(evs)
        for s, e in truth:
            hit = np.flatnonzero(p[s:e])
            if len(hit):
                c["detected"] += 1
                c["delays_ms"].append(float(hit[0]) * 1000.0 / rate_hz)
        for s, e in evs:
            if t[s:e].mean() >= MIN_EVENT_PRECISION:
                c["correct"] += 1
            else:
                c["false_alarms"] += 1
    return c


def merge_counts(cs: Sequence[Dict]) -> Dict:
    out = dict(n_true=0, detected=0, n_pred=0, correct=0, false_alarms=0, frames=0,
               tp=0, fn=0, fp=0, tn=0, delays_ms=[])
    for c in cs:
        for k in out:
            out[k] = out[k] + c[k] if k != "delays_ms" else out[k] + list(c[k])
    return out


def summarize_counts(c: Dict, rate_hz: float) -> Dict[str, float]:
    nan = float("nan")
    prec = c["correct"] / c["n_pred"] if c["n_pred"] else (nan if c["n_true"] == 0 else 0.0)
    rec = c["detected"] / c["n_true"] if c["n_true"] else nan
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec > 0 else (
        0.0 if prec == prec and rec == rec else nan)
    minutes = c["frames"] / rate_hz / 60.0
    return {
        "event_precision": prec * 100, "event_recall": rec * 100, "event_f1": f1 * 100,
        "sensitivity": 100.0 * c["tp"] / max(c["tp"] + c["fn"], 1),
        "specificity": 100.0 * c["tn"] / max(c["tn"] + c["fp"], 1),
        "frame_acc": 100.0 * (c["tp"] + c["tn"]) / max(c["frames"], 1),
        "false_alarms_per_min": c["false_alarms"] / max(minutes, 1e-9),
        "missed_freezes": c["n_true"] - c["detected"], "false_alarms": c["false_alarms"],
        "n_episodes": c["n_true"],
        "median_delay_ms": float(np.median(c["delays_ms"])) if c["delays_ms"] else nan,
        "freeze_frac": 100.0 * (c["tp"] + c["fn"]) / max(c["frames"], 1),
    }


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """PR-AUC as average precision (step-wise integral of precision over recall)."""
    labels = labels.astype(bool)
    npos = int(labels.sum())
    if npos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = np.cumsum(labels[order])
    prec = tp / np.arange(1, len(tp) + 1)
    # only count positions at distinct score values
    distinct = np.r_[np.flatnonzero(np.diff(scores[order])), len(scores) - 1]
    tp_d, prec_d = tp[distinct], prec[distinct]
    rec_d = tp_d / npos
    return float(np.sum(np.diff(np.r_[0.0, rec_d]) * prec_d))


def select_threshold(probs: Sequence[np.ndarray], trues: Sequence[np.ndarray], rate_hz: float,
                     grid: Sequence[float] = THRESHOLDS) -> float:
    """Decision threshold that maximises pooled event F1 on VALIDATION data (frame F1 breaks ties)."""
    best, best_thr = -1.0, float(grid[len(grid) // 2])
    for thr in grid:
        c = event_counts([p >= thr for p in probs], trues, rate_hz)
        s = summarize_counts(c, rate_hz)
        f1 = 0.0 if s["event_f1"] != s["event_f1"] else s["event_f1"]
        fr = 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)
        score = f1 + 1e-3 * fr
        if score > best:
            best, best_thr = score, float(thr)
    return best_thr


# ---------------------------------------------------------------------------------------
# Windowed training data
# ---------------------------------------------------------------------------------------
def make_windows(trials: Sequence[Trial], norm: Normalizer, win: int, stride: int):
    xs, ys, ms = [], [], []
    for t in trials:
        x, y = norm(t.kinematics), t.gestures
        starts = list(range(0, max(len(x) - win, 0) + 1, stride))
        if len(x) > win and starts[-1] + win < len(x):
            starts.append(len(x) - win)
        for s in starts:
            xw, yw = x[s:s + win], y[s:s + win]
            n = len(xw)
            X = np.zeros((win, x.shape[1]), np.float32); X[:n] = xw
            Y = np.full(win, -100, np.int64); Y[:n] = yw
            M = np.zeros(win, bool); M[:n] = True
            xs.append(X); ys.append(Y); ms.append(M)
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)), torch.from_numpy(np.stack(ms))


@torch.no_grad()
def predict_logits(model: SeqModel, trials_x: Sequence[np.ndarray], device: str = "cpu") -> List[np.ndarray]:
    """Offline pass over each full trial -> logits [T, 2]. For causal models this equals the
    streaming output (verified by tests)."""
    model.eval()
    return [model(torch.from_numpy(x).unsqueeze(0).to(device))[0][0].cpu().numpy() for x in trials_x]


@torch.no_grad()
def predict_logits_online_acausal(model: SeqModel, trials_x: Sequence[np.ndarray], hist: int,
                                  stride: int, device: str = "cpu", chunk: int = 256) -> List[np.ndarray]:
    """What a deployed acausal model can do: every ``stride`` frames, run it on the last ``hist``
    frames only and keep the answer for the newest frame until the next recomputation."""
    model.eval()
    outs = []
    for x in trials_x:
        T = len(x)
        points = list(range(0, T, stride))
        logits = np.zeros((T, 2), np.float32)
        for c0 in range(0, len(points), chunk):
            pts = points[c0:c0 + chunk]
            lens = [min(t + 1, hist) for t in pts]
            X = torch.zeros(len(pts), hist, x.shape[1])
            M = torch.zeros(len(pts), hist, dtype=torch.bool)
            for i, (t, n) in enumerate(zip(pts, lens)):
                X[i, :n] = torch.from_numpy(x[t + 1 - n:t + 1]); M[i, :n] = True
            g, _ = model(X.to(device), M.to(device))
            last = g[torch.arange(len(pts)), torch.tensor(lens) - 1].cpu().numpy()
            for t, l in zip(pts, last):
                logits[t:t + stride] = l
        outs.append(logits)
    return outs


def freeze_prob(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(-1))


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------
def train_fog_model(model: SeqModel, train_trials, val_trials, norm: Normalizer, rate_hz: float,
                    epochs: int, seed: int, win: int = 512, stride: int = 256, bs: int = 32,
                    lr: float = 2e-3, device: str = "cpu", eval_every: int = 2):
    """Weighted cross-entropy on random windows. The epoch with the best validation PR-AUC is kept
    (validation subjects only; the test subject is never touched)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    X, Y, M = make_windows(train_trials, norm, win, stride)
    n1 = max(int((Y == 1).sum()), 1); n0 = max(int((Y == 0).sum()), 1)
    w = torch.tensor([1.0, float(min(np.sqrt(n0 / n1), 10.0))], device=device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    val_x = [norm(t.kinematics) for t in val_trials]
    val_y = np.concatenate([t.gestures for t in val_trials])
    best, best_state, hist = -1.0, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb, yb, mb = X[idx].to(device), Y[idx].to(device), M[idx].to(device)
            logits, _ = model(xb, mb)
            loss = F.cross_entropy(logits.reshape(-1, 2), yb.reshape(-1), weight=w, ignore_index=-100)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(idx)
        sched.step()
        rec = {"epoch": ep, "train_loss": tot / len(X)}
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            ap = average_precision(np.concatenate([freeze_prob(l) for l in predict_logits(model, val_x, device)]),
                                   val_y)
            rec["val_pr_auc"] = ap
            if ap > best:
                best = ap; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist.append(rec)
    model.load_state_dict(best_state)
    model.eval()
    return hist, best


# ---------------------------------------------------------------------------------------
# Freeze Index detector
# ---------------------------------------------------------------------------------------
def fi_scores(trials: Sequence[Trial], rate_hz: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    out = [freeze_index(t.kinematics[:, ANKLE_VERTICAL], rate_hz) for t in trials]
    return [o[0] for o in out], [o[1] for o in out]


def select_fi_thresholds(val_trials, rate_hz: float) -> Tuple[float, float]:
    """(FI threshold, power threshold) maximising pooled event F1 on VALIDATION subjects."""
    fi, pw = fi_scores(val_trials, rate_hz)
    trues = [t.gestures for t in val_trials]
    allfi, allpw = np.concatenate(fi), np.concatenate(pw)
    fi_grid = np.unique(np.quantile(allfi[allpw > 0], np.linspace(0.3, 0.995, 40)))
    pw_grid = np.unique(np.r_[0.0, np.quantile(allpw[allpw > 0], [0.1, 0.25, 0.5, 0.75])])
    best, best_pair = -1.0, (float(fi_grid[0]), 0.0)
    for th in fi_grid:
        for pt in pw_grid:
            c = event_counts([(f > th) & (p > pt) for f, p in zip(fi, pw)], trues, rate_hz)
            s = summarize_counts(c, rate_hz)
            f1 = 0.0 if s["event_f1"] != s["event_f1"] else s["event_f1"]
            score = f1 + 1e-3 * 2 * c["tp"] / max(2 * c["tp"] + c["fp"] + c["fn"], 1)
            if score > best:
                best, best_pair = score, (float(th), float(pt))
    return best_pair
