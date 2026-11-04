"""Tables and figures for calibration, robustness and abstention (from ``robustness.json``)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fog import merge_counts, summarize_counts
from .fog_report import COLORS
from .fog_run import FI_NAME
from .robustness import CONDITIONS, cond_name


def _ms(v, d=1):
    v = np.asarray([x for x in v if x == x], float)
    return "n/a" if not len(v) else f"{v.mean():.{d}f} ± {v.std():.{d}f}"


def _pooled(rows, rate, **flt):
    """Per-seed pooled summaries for the rows matching ``flt``."""
    sel = [r for r in rows if all(r.get(k) == v for k, v in flt.items())]
    out = []
    for s in sorted({r["seed"] for r in sel}):
        c = merge_counts([r["counts"] for r in sel if r["seed"] == s])
        d = summarize_counts(c, rate)
        if "covered" in sel[0]["counts"]:
            d["coverage"] = 100.0 * sum(r["counts"]["covered"] for r in sel if r["seed"] == s) / max(
                sum(r["counts"]["total"] for r in sel if r["seed"] == s), 1)
            d["covered_acc"] = 100.0 * sum(r["counts"]["cov_correct"] for r in sel if r["seed"] == s) / max(
                sum(r["counts"]["covered"] for r in sel if r["seed"] == s), 1)
        d["detected"], d["n_true"] = c["detected"], c["n_true"]
        out.append(d)
    return out


def write_robustness_report(res: Dict, out: Path) -> None:
    res = json.loads(json.dumps(res, default=float))
    out = Path(out); fig = out / "figures"; fig.mkdir(parents=True, exist_ok=True)
    rate, models = res["meta"]["rate_hz"], res["meta"]["models"]
    cal, ab, fl = res["calibration"], res["abstention"], res["faults"]
    taus, names = res["meta"]["taus"], [FI_NAME] + models
    L = ["# Calibration, abstention and sensor-fault robustness (Daphnet, real data)", "",
         f"Seeds {res['meta']['seeds']} · models loaded from the main benchmark (no retraining). The "
         "temperature, the decision threshold and the abstention threshold were all chosen on the validation "
         "subjects of each fold, on clean data. Faults are applied to the raw test signals only. Event metrics "
         "are pooled over test subjects within a seed (mean ± std across seeds).", "",
         "## Calibration (temperature scaling fitted on validation subjects)", "",
         "ECE uses 15 equal-width bins on the test subjects (mean ± std across seeds and folds). "
         "\"Freeze ECE\" is the calibration of p(freeze) itself.", "",
         "| Model | Temperature T | ECE before | ECE after | Freeze ECE before | Freeze ECE after | NLL before | NLL after |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = [x for x in cal if x["model"] == m]
        L.append(f"| {m} | {_ms([x['T'] for x in r], 2)} | " + " | ".join(
            _ms([x[k] * 100 if "ece" in k else x[k] for x in r], 2 if "ece" in k else 3)
            for k in ("ece_before", "ece_after", "freeze_ece_before", "freeze_ece_after", "nll_before", "nll_after")) + " |")
    L += ["", "(ECE in percent.)", "",
          "## Abstention: coverage vs. event-level performance (clean test data)", "",
          "A frame is \"uncertain\" when max(p, 1-p) < tau after temperature scaling; uncertain frames raise no "
          "alarm. Coverage is the fraction of frames with a decision.", "",
          "| Model | tau | Coverage (%) | Accuracy on covered frames (%) | Event F1 | Event recall | False alarms / min | Missed freezes | False alarms |",
          "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        for tau in taus:
            ps = _pooled([r for r in ab], rate, model=m, tau=tau)
            L.append(f"| {m} | {tau} | {_ms([p['coverage'] for p in ps])} | {_ms([p['covered_acc'] for p in ps])} | "
                     f"{_ms([p['event_f1'] for p in ps])} | {_ms([p['event_recall'] for p in ps])} | "
                     f"{_ms([p['false_alarms_per_min'] for p in ps], 2)} | "
                     f"{np.mean([p['missed_freezes'] for p in ps]):.0f} | {np.mean([p['false_alarms'] for p in ps]):.0f} |")
    star = {m: Counter(x["tau_star"] for x in cal if x["model"] == m).most_common(1)[0][0] for m in models}
    L += ["", "Operating abstention threshold (best validation event F1 with coverage >= "
          f"{int(res['meta']['min_coverage'] * 100)}%; the most frequent choice over folds and seeds is shown, "
          "each fold uses its own): " + ", ".join(f"{m} {star[m]}" for m in models) + ".", ""]
    L += ["## Sensor-fault robustness (online recognition, fixed thresholds from clean validation data)", "",
          "Dropped frames hold the last received frame. Noise is Gaussian with std = factor x each feature's "
          "training-set std. A stuck sensor repeats its value for the given number of seconds (one event per 30 s "
          "of recording at random times). \"Missed\" = freeze episodes never detected; \"FA\" = false alarms "
          "(counts summed over the 8 test subjects, mean over seeds; the clean test set has "
          f"{_pooled(fl, rate, model=FI_NAME, kind='clean', mode='plain')[0]['n_true']} episodes).", ""]
    hdr = "| Condition | " + " | ".join(names) + " |"
    for metric, label, key, d in (("event_f1", "Event F1", None, 1), ("event_recall", "Event recall (%)", None, 1),
                                  ("false_alarms_per_min", "False alarms per minute", None, 2),
                                  ("frame_acc", "Frame accuracy (%)", None, 1)):
        L += [f"### {label}", "", hdr, "|---|" + "---|" * len(names)]
        for kind, level in CONDITIONS:
            cells = [_ms([p[metric] for p in _pooled(fl, rate, model=m, kind=kind, level=level, mode="plain")], d)
                     for m in names]
            L.append(f"| {cond_name(kind, level)} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["### Missed freezes vs. false alarms (counts; plain decision / with abstention)", "",
          "| Condition | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for kind, level in CONDITIONS:
        cells = []
        for m in names:
            a = _pooled(fl, rate, model=m, kind=kind, level=level, mode="plain")
            b = _pooled(fl, rate, model=m, kind=kind, level=level, mode="abstain") if m != FI_NAME else None
            f = lambda ps: f"{np.mean([p['missed_freezes'] for p in ps]):.0f} / {np.mean([p['false_alarms'] for p in ps]):.0f}"
            cells.append(f(a) + (f" ; {f(b)}" if b else ""))
        L.append(f"| {cond_name(kind, level)} | " + " | ".join(cells) + " |")
    L += ["", "Each cell: missed / false alarms."]
    (out / "robustness.md").write_text("\n".join(L) + "\n")

    # ---- robustness.png: freeze event F1 vs fault severity ------------------------------------
    panels = [("drop", "Dropped frames (%)", [0, 5, 10, 20]), ("noise", "Noise (x feature std)", [0, 0.25, 0.5, 1.0]),
              ("freeze", "Sensor freeze (s)", [0, 1, 2, 3])]
    f1, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (kind, xl, xs) in zip(axes, panels):
        levels = [l for k, l in CONDITIONS if k == kind]
        for m in names:
            y, e = [], []
            for kk, lv in [("clean", 0.0)] + [(kind, l) for l in levels]:
                v = [p["event_f1"] for p in _pooled(fl, rate, model=m, kind=kk, level=lv, mode="plain")]
                y.append(np.mean(v)); e.append(np.std(v))
            ax.errorbar(xs, y, yerr=e, marker="o", label=m, color=COLORS[m], capsize=2,
                        ls="--" if m == FI_NAME else "-")
        ax.set_xlabel(xl); ax.set_xticks(xs)
    axes[0].set_ylabel("Freeze event F1 (%)"); axes[0].legend(frameon=False, fontsize=8)
    f1.suptitle("Online freeze detection under sensor faults (Daphnet, mean ± std over seeds)", fontsize=10)
    f1.tight_layout(); f1.savefig(fig / "robustness.png", dpi=160); plt.close(f1)

    # ---- abstention.png -------------------------------------------------------------------
    f2, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    for m in models:
        cov, f1v, fa = [], [], []
        for tau in taus:
            ps = _pooled(ab, rate, model=m, tau=tau)
            cov.append(np.mean([p["coverage"] for p in ps])); f1v.append(np.mean([p["event_f1"] for p in ps]))
            fa.append(np.mean([p["false_alarms_per_min"] for p in ps]))
        a1.plot(cov, f1v, marker="o", label=m, color=COLORS[m]); a2.plot(cov, fa, marker="o", color=COLORS[m])
        for c_, f_, t_ in zip(cov, f1v, taus):
            a1.annotate(f"{t_}", (c_, f_), fontsize=6, xytext=(2, 2), textcoords="offset points")
    a1.set_xlabel("Coverage (% of frames with a decision)"); a1.set_ylabel("Event F1 (%)")
    a2.set_xlabel("Coverage (% of frames with a decision)"); a2.set_ylabel("False alarms per minute")
    a1.legend(frameon=False, fontsize=8); f2.suptitle("Abstention after temperature scaling (labels: tau)", fontsize=10)
    f2.tight_layout(); f2.savefig(fig / "abstention.png", dpi=160); plt.close(f2)
    print(f"[done] {out / 'robustness.md'}, {fig / 'robustness.png'}, {fig / 'abstention.png'}")
