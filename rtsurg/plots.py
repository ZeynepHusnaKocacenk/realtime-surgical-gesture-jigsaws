"""Portfolio figures. Colour follows the model (fixed order), and causal vs. acausal is
encoded a second way (filled vs. hollow markers / solid vs. dashed lines), so the
charts never rely on colour alone."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"TCN-causal": "#2a78d6", "TCN-acausal": "#eb6834", "LSTM": "#1baf7a", "BiLSTM": "#eda100"}
CAUSAL = {"TCN-causal": True, "LSTM": True, "TCN-acausal": False, "BiLSTM": False}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
})


def _label(m: str) -> str:
    return f"{m} ({'causal' if CAUSAL.get(m) else 'acausal'})"


def accuracy_vs_latency(summary: Dict, frame_budget_ms: float, ref_history: int, path: Path,
                        ref_seconds: float = 60):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, r in summary.items():
        lat = r["latency"]["by_history"][ref_history]["p95_ms"]
        acc = r["online"]["frame_acc"]["mean"]
        ax.scatter(lat, acc, s=90, color=COLORS[m] if CAUSAL[m] else SURFACE,
                   edgecolors=COLORS[m], linewidths=2, zorder=3, label=_label(m))
        ax.annotate(m, (lat, acc), xytext=(8, 4), textcoords="offset points", color=INK, fontsize=9)
    ax.axvline(frame_budget_ms, color=INK2, linestyle=":", linewidth=1.2)
    ax.text(frame_budget_ms, ax.get_ylim()[0], f"  frame budget {frame_budget_ms:.0f} ms",
            color=INK2, fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel(f"p95 latency per update (ms, log scale), {ref_seconds:.0f} s into the procedure")
    ax.set_ylabel("Online frame accuracy (%)")
    ax.set_title("Real-time gesture recognition: accuracy vs. latency")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def offline_vs_online(summary: Dict, path: Path):
    metrics = [("frame_acc", "Frame acc."), ("edit", "Edit"), ("f1@25", "F1@25")]
    models = list(summary)
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 3.8), sharey=True)
    x = np.arange(len(models))
    w = 0.38
    for ax, (key, title) in zip(axes, metrics):
        for j, mode in enumerate(("offline", "online")):
            vals = [summary[m][mode][key]["mean"] for m in models]
            errs = [summary[m][mode][key]["std"] for m in models]
            ax.bar(x + (j - 0.5) * w, vals, w - 0.03, yerr=errs, capsize=2,
                   color=[COLORS[m] for m in models], alpha=1.0 if mode == "offline" else 0.45,
                   hatch=None if mode == "offline" else "//", edgecolor=SURFACE, linewidth=0,
                   error_kw={"ecolor": INK2, "elinewidth": 0.8})
        ax.set_xticks(x, models, rotation=20, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Score (%)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=INK2, label="offline (whole trial seen)"),
                        Patch(facecolor=INK2, alpha=0.45, hatch="//", label="online (past frames only)")],
               loc="upper right", fontsize=8)
    fig.suptitle("What each model loses when it cannot see the future", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def latency_vs_history(summary: Dict, frame_ms: float, frame_budget_ms: float, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, r in summary.items():
        hs = sorted(r["latency"]["by_history"])
        secs = [h * frame_ms / 1000 for h in hs]
        ys = [r["latency"]["by_history"][h]["median_ms"] for h in hs]
        ax.plot(secs, ys, color=COLORS[m], linewidth=2, linestyle="-" if CAUSAL[m] else "--",
                marker="o", markersize=5, label=_label(m))
    ax.axhline(frame_budget_ms, color=INK2, linestyle=":", linewidth=1.2)
    ax.text(ax.get_xlim()[0], frame_budget_ms, f" frame budget {frame_budget_ms:.0f} ms", color=INK2,
            fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("Elapsed procedure time (s)")
    ax.set_ylabel("Median latency per update (ms, log)")
    ax.set_title("Causal models stay flat; acausal re-runs grow with time")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def early_skill(summary: Dict, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m, r in summary.items():
        ps = sorted(r["early_skill"])
        ax.plot([p * 100 for p in ps], [r["early_skill"][p]["mean"] for p in ps], color=COLORS[m],
                linewidth=2, linestyle="-" if CAUSAL[m] else "--", marker="o", markersize=5, label=_label(m))
    ax.set_xlabel("% of trial observed")
    ax.set_ylabel("Skill accuracy, LOUO (%)")
    ax.set_ylim(0, 105)
    ax.set_title("How early can skill level be called?")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def gesture_timeline(example: Dict[str, Dict[str, List[int]]], gesture_names: List[str],
                     frame_ms: float, path: Path):
    """Ground truth vs. online predictions of each model on one held-out trial."""
    rows = [("Ground truth", example[next(iter(example))]["true"])]
    rows += [(f"{m} online", ex["online"]) for m, ex in example.items()]
    cmap = plt.get_cmap("tab20", len(gesture_names))
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 0.55 * len(rows) + 1.2), sharex=True)
    for ax, (name, seq) in zip(axes, rows):
        seq = np.asarray(seq)
        ax.imshow(seq[None, :], aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(gesture_names) - 0.5,
                  interpolation="nearest", extent=(0, len(seq) * frame_ms / 1000, 0, 1))
        ax.set_yticks([]); ax.grid(False)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[-1].set_xlabel("Time (s)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=cmap(i), label=g) for i, g in enumerate(gesture_names)],
               loc="lower center", ncol=min(len(gesture_names), 10), fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Online gesture segmentation on a held-out surgeon", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
