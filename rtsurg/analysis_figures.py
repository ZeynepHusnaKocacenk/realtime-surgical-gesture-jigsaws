"""Analysis figures for the README and the project page, all computed from saved result files.

    python -m rtsurg.analysis_figures          # writes docs/figures/analysis/*.png

Colour roles are fixed across every figure (an entity keeps its colour; the Freeze Index is the
neutral reference): palette from the dataviz reference instance, validated for colour-vision
deficiency; every line is direct-labelled so identity never relies on colour alone.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fog import merge_counts, summarize_counts

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "docs" / "figures" / "analysis"
RATE = 32.0
FI = "Freeze Index"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {FI: "#7d7c78", "TCN-causal": "#2a78d6", "TCN-spectral": "#eb6834", "TCN-spectral + noise aug.": "#1baf7a"}
RUN = {"TCN-causal": "daphnet", "TCN-spectral": "daphnet_spectral", "TCN-spectral + noise aug.": "daphnet_spectral_aug",
       "TCN-spectral, tuned": "daphnet_tuned"}
COL["TCN-spectral, tuned"] = COL["TCN-spectral"]   # the selected model always wears slot 2 (orange);
# figures never show both TCN-spectral variants together (all-pairs CVD rule: 3 coloured series max)


def selected() -> str:
    """The learned model chosen on validation patients (rtsurg/fog_summary.py)."""
    return json.load(open(RES / "daphnet_summary.json"))["selected_by_validation"]

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def _rows(run: str, model: str, mode: str = "online") -> List[Dict]:
    res = json.load(open(RES / run / "results.json"))
    return [r for r in res["rows"] if r["model"] == model and r["mode"] == mode]


def _detector_rows(name: str, mode: str = "online") -> List[Dict]:
    return _rows("daphnet", FI, mode) if name == FI else _rows(RUN[name], "TCN-causal", mode)


def _per_patient(rows, fn) -> Dict[str, float]:
    out: Dict[str, List[float]] = {}
    for r in rows:
        v = fn(r)
        out.setdefault(r["fold"], []).append(0.0 if v != v else v)
    return {k: float(np.mean(v)) for k, v in sorted(out.items())}


def per_patient_paired():
    a, b = FI, selected()
    metrics = [("Online event F1 (%)", lambda r: r["summary"]["event_f1"]), ("PR-AUC", lambda r: r["pr_auc"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
    for ax, (label, fn) in zip(axes, metrics):
        pa, pb = _per_patient(_detector_rows(a), fn), _per_patient(_detector_rows(b), fn)
        pts = list(pa)
        for i, p in enumerate(pts):
            up = pb[p] > pa[p]
            ax.plot([i, i], [pa[p], pb[p]], color=COL[b] if up else INK2, lw=1.5, alpha=0.55, zorder=1)
        ax.scatter(range(len(pts)), [pa[p] for p in pts], s=46, color=COL[a], zorder=3, label=a,
                   edgecolor=SURF, linewidth=1.5)
        ax.scatter(range(len(pts)), [pb[p] for p in pts], s=46, color=COL[b], zorder=3, label=b,
                   edgecolor=SURF, linewidth=1.5)
        wins = sum(pb[p] > pa[p] for p in pts)
        ax.set_xticks(range(len(pts))); ax.set_xticklabels(pts)
        ax.set_title(f"{label}: {b} higher in {wins}/{len(pts)} patients", loc="left")
        ax.set_xlabel("held-out patient (mean over 3 seeds)")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Per-patient comparison with the Freeze Index (each patient never seen in training or selection)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "per_patient_paired.png", dpi=170); plt.close(fig)


def model_selection():
    s = json.load(open(RES / "daphnet_summary.json"))
    names = list(s["validation_pr_auc"])
    val = [s["validation_pr_auc"][n] for n in names]
    test = [s["detectors"][f"{n} | threshold"]["pr_auc"]["point"] for n in names]
    lo = [s["detectors"][f"{n} | threshold"]["pr_auc"]["lo"] for n in names]
    hi = [s["detectors"][f"{n} | threshold"]["pr_auc"]["hi"] for n in names]
    order = np.argsort(val)
    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    for k, i in enumerate(order):
        sel = names[i] == s["selected_by_validation"]
        ax.plot([val[i], test[i]], [k, k], color=GRID, lw=2, zorder=1)
        ax.errorbar(test[i], k, xerr=[[test[i] - lo[i]], [hi[i] - test[i]]], fmt="none", ecolor=INK2, lw=1, capsize=3, zorder=2)
        ax.scatter(val[i], k, s=60, marker="D", color="#6250d6", zorder=3, edgecolor=SURF, linewidth=1.5,
                   label="validation patients (used to select)" if k == 0 else None)
        ax.scatter(test[i], k, s=60, color="#2a78d6", zorder=3, edgecolor=SURF, linewidth=1.5,
                   label="held-out test patients, 95% CI" if k == 0 else None)
        ax.text(max(val[i], hi[i]) + 0.012, k, ("selected  " if sel else "") + f"val {val[i]:.3f} / test {test[i]:.3f}",
                va="center", fontsize=8, color=INK if sel else INK2, fontweight="bold" if sel else "normal")
    fi = s["detectors"]["Freeze Index | threshold"]["pr_auc"]["point"]
    ax.axvline(fi, color=COL[FI], ls=":", lw=1.2)
    ax.text(fi, -0.55, " Freeze Index (test)", color=INK2, fontsize=8, va="center")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names[i] for i in order])
    ax.set_xlabel("PR-AUC"); ax.set_xlim(0.28, 0.98); ax.set_ylim(-0.9, len(order) - 0.5); ax.grid(axis="y", visible=False)
    ax.set_title("Model selection uses validation patients only; test values shown for transparency", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "model_selection.png", dpi=170); plt.close(fig)


def _fault_curve(run: str, model: str, kind: str, metric: str):
    r = json.load(open(RES / run / "robustness.json"))["faults"]
    levels = sorted({x["level"] for x in r if x["kind"] == kind})
    out = []
    for lv in [0.0] + levels:
        k = "clean" if lv == 0.0 else kind
        sel = [x for x in r if x["model"] == model and x["kind"] == k and (lv == 0.0 or x["level"] == lv)
               and x["mode"] == "plain"]
        per_seed = [summarize_counts(merge_counts([x["counts"] for x in sel if x["seed"] == sd]), RATE)[metric]
                    for sd in sorted({x["seed"] for x in sel})]
        out.append((lv, float(np.mean(per_seed)), float(np.std(per_seed))))
    return out


def robustness_noise():
    series = [(FI, "daphnet", FI), ("TCN-causal", "daphnet", "TCN-causal"),
              (selected(), RUN[selected()], "TCN-causal"),
              ("TCN-spectral + noise aug.", "daphnet_spectral_aug", "TCN-causal")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, (metric, label) in zip(axes, [("event_f1", "Event F1 (%)"), ("false_alarms_per_min", "False alarms per minute")]):
        ends = []
        for name, run, model in series:
            c = _fault_curve(run, model, "noise", metric)
            x = [v[0] for v in c]; y = [v[1] for v in c]; e = [v[2] for v in c]
            ax.plot(x, y, color=COL[name], lw=2, marker="o", ms=5, mec=SURF, mew=1.2)
            ax.fill_between(x, np.subtract(y, e), np.add(y, e), color=COL[name], alpha=0.10, lw=0)
            ends.append([y[-1], name])
        ends.sort()                                     # direct labels at the right end, nudged apart
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.07 * span)
        for y, name in ends:
            ax.text(1.03, y, name, color=INK, fontsize=8, va="center")
            ax.plot([1.0, 1.025], [y, y], color=COL[name], lw=2)
        ax.set_xlim(-0.03, 1.55); ax.set_xticks([0, 0.25, 0.5, 1.0])
        ax.set_xlabel("Gaussian sensor noise (x feature std, test patients only)")
        ax.set_title(label, loc="left")
    fig.suptitle("Sensor noise is the main failure mode; noise augmentation did not fix it (mean ± std over 3 seeds)",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "robustness_noise.png", dpi=170); plt.close(fig)


def onset_delay():
    names = [FI, "TCN-causal", selected()]
    data = [np.asarray([d for r in _detector_rows(n) for d in r["counts"]["delays_ms"]]) / 1000 for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    rng = np.random.default_rng(0)
    for i, (n, d) in enumerate(zip(names, data)):
        y = i + rng.uniform(-0.18, 0.18, len(d))
        ax.scatter(np.clip(d, 0, 6), y, s=7, color=COL[n], alpha=0.35, lw=0)
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=INK, lw=3, solid_capstyle="round")
        ax.scatter([med], [i], s=60, color=SURF, edgecolor=INK, linewidth=1.8, zorder=4)
        ax.text(6.15, i, f"median {med * 1000:.0f} ms  (n = {len(d)})", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlim(-0.1, 8.6); ax.set_xticks(range(0, 7)); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Time from annotated freeze onset to first alarm inside it (s; detected episodes, 3 seeds, clipped at 6 s)")
    ax.set_title("How fast each detector flags a freeze (bar = interquartile range)", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "onset_delay.png", dpi=170); plt.close(fig)


def operating_points():
    s = json.load(open(RES / "daphnet_summary.json"))["detectors"]
    names = [FI, "TCN-causal", selected(), "TCN-spectral + noise aug."]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for n in names:
        d = s[f"{n} | threshold"]
        ax.errorbar(d["false_alarms_per_min"]["point"], d["event_recall"]["point"],
                    xerr=[[d["false_alarms_per_min"]["point"] - d["false_alarms_per_min"]["lo"]],
                          [d["false_alarms_per_min"]["hi"] - d["false_alarms_per_min"]["point"]]],
                    yerr=[[d["event_recall"]["point"] - d["event_recall"]["lo"]],
                          [d["event_recall"]["hi"] - d["event_recall"]["point"]]],
                    fmt="o", ms=9, color=COL[n], mec=SURF, mew=1.5, ecolor=COL[n], elinewidth=1.2, capsize=3)
        off = (8, -14) if n == "TCN-spectral + noise aug." else (8, 6)
        ax.annotate(n, (d["false_alarms_per_min"]["point"], d["event_recall"]["point"]), xytext=off,
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("False alarms per minute (lower is better)"); ax.set_ylabel("Episodes detected (%)")
    ax.set_title("Operating points at the validation-chosen threshold (95% CI over patients)", loc="left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "operating_points.png", dpi=170); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (per_patient_paired, model_selection, robustness_noise, onset_delay, operating_points):
        fn(); print(f"[figure] {fn.__name__}")


if __name__ == "__main__":
    main()
