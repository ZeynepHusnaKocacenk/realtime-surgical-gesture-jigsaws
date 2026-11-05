"""Uncertainty for the Daphnet results: subject-level bootstrap CIs and paired tests.

The unit of independence is the TEST SUBJECT (8 of them), not the frame or the episode: frames and
episodes of one patient are strongly correlated, so resampling them would give intervals that are
far too narrow. Each bootstrap replicate draws 8 subjects with replacement, pools their event
counts within each seed (exactly as the main table does), computes the metric per seed and averages
over seeds. A paired comparison of two detectors uses the same resampled subjects for both, so
between-subject variation (the dominant source of variance here) cancels.

The exact Wilcoxon signed-rank test on per-subject differences (mean over seeds) is reported next
to it. With 8 subjects its smallest possible two-sided p-value is 2/256 = 0.0078, so "not
significant" here mostly means "8 patients cannot tell"; these are descriptive statistics for a
small cohort, not a confirmatory trial.
"""
from __future__ import annotations

from itertools import product
from typing import Callable, Dict, List, Sequence

import numpy as np

from .fog import merge_counts, summarize_counts


def _rows(rows, model, mode):
    return [r for r in rows if r["model"] == model and r["mode"] == mode]


def _table(rows, model, mode, key):
    """{seed: {fold: counts}}"""
    t: Dict = {}
    for r in _rows(rows, model, mode):
        t.setdefault(r["seed"], {})[r["fold"]] = r[key]
    return t


def _pooled_metric(table, folds: Sequence[str], metric: str, rate: float) -> float:
    vals = []
    for seed, by_fold in table.items():
        s = summarize_counts(merge_counts([by_fold[f] for f in folds]), rate)[metric]
        vals.append(0.0 if s != s else s)
    return float(np.mean(vals))


def bootstrap_metric(rows, model: str, mode: str, rate: float, metric: str = "event_f1",
                     key: str = "counts", n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Point estimate (all subjects) and 95% percentile CI over resampled test subjects."""
    table = _table(rows, model, mode, key)
    folds = sorted(next(iter(table.values())))
    rng = np.random.default_rng(seed)
    point = _pooled_metric(table, folds, metric, rate)
    boots = [_pooled_metric(table, list(rng.choice(folds, len(folds))), metric, rate) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "lo": float(lo), "hi": float(hi)}


def per_subject(rows, model: str, mode: str, value: Callable[[Dict], float]) -> Dict[str, float]:
    """Per-subject value averaged over seeds."""
    out: Dict[str, List[float]] = {}
    for r in _rows(rows, model, mode):
        v = value(r)
        if v == v:
            out.setdefault(r["fold"], []).append(v)
    return {f: float(np.mean(v)) for f, v in out.items()}


def bootstrap_mean(values: Dict[str, float], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    folds = sorted(values)
    v = np.array([values[f] for f in folds])
    rng = np.random.default_rng(seed)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return {"point": float(v.mean()), "lo": float(lo), "hi": float(hi)}


def paired_mean_diff(a: Dict[str, float], b: Dict[str, float], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Mean over subjects of a[s] - b[s] (per-subject values, e.g. PR-AUC): point, paired-bootstrap
    95% CI, P(>0), exact Wilcoxon p and wins."""
    folds = sorted(set(a) & set(b))
    d = np.array([a[f] - b[f] for f in folds])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_boot_gt0": float((boots > 0).mean()),
            "wilcoxon_p": wilcoxon_exact(d), "wins": int((d > 0).sum()), "n_subjects": len(folds)}


def wilcoxon_exact(diffs: Sequence[float]) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value (zeros dropped, average ranks for ties)."""
    d = np.asarray([x for x in diffs if abs(x) > 1e-12], float)
    n = len(d)
    if n == 0:
        return 1.0
    a = np.abs(d)
    order = a.argsort()
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(a):                                   # average ranks of ties
        ranks[a == v] = ranks[a == v].mean()
    w_obs = ranks[d > 0].sum()
    mu = ranks.sum() / 2
    ws = np.array([sum(r for r, s in zip(ranks, signs) if s) for signs in product([0, 1], repeat=n)])
    return float(np.mean(np.abs(ws - mu) >= abs(w_obs - mu) - 1e-9))


def paired_comparison(rows, a: str, b: str, rate: float, mode_a: str = "online", mode_b: str = "online",
                      metric: str = "event_f1", key: str = "counts", n_boot: int = 2000,
                      seed: int = 0) -> Dict[str, float]:
    """Difference a - b in the pooled metric: point, 95% paired-bootstrap CI, bootstrap
    P(diff > 0), and the exact Wilcoxon p-value on per-subject values (mean over seeds)."""
    ta, tb = _table(rows, a, mode_a, key), _table(rows, b, mode_b, key)
    folds = sorted(next(iter(ta.values())))
    rng = np.random.default_rng(seed)
    point = _pooled_metric(ta, folds, metric, rate) - _pooled_metric(tb, folds, metric, rate)
    boots = []
    for _ in range(n_boot):
        fs = list(rng.choice(folds, len(folds)))
        boots.append(_pooled_metric(ta, fs, metric, rate) - _pooled_metric(tb, fs, metric, rate))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    def subj(t):
        out = {}
        for f in folds:
            v = [summarize_counts(t[s][f], rate)[metric] for s in t]
            out[f] = float(np.mean([0.0 if x != x else x for x in v]))
        return out
    sa, sb = subj(ta), subj(tb)
    return {"diff": float(point), "lo": float(lo), "hi": float(hi),
            "p_boot_gt0": float(np.mean(np.asarray(boots) > 0)),
            "wilcoxon_p": wilcoxon_exact([sa[f] - sb[f] for f in folds]),
            "wins": int(sum(sa[f] > sb[f] for f in folds)), "n_subjects": len(folds)}
