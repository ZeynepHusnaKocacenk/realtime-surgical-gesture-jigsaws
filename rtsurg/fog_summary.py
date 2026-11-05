"""Headline Daphnet table with uncertainty: merges the baseline run, the improved model and the
ablations, and writes ``results/daphnet_summary.md`` + ``.json``.

    python -m rtsurg.fog_summary

Every number is computed from the saved ``results.json`` files (no retraining). Each detector is
listed at two operating points, both chosen on validation subjects only:
  * ``threshold``          the original protocol (decision threshold);
  * ``threshold+debounce`` threshold and a minimum alarm duration (0 to 1.5 s), for every detector,
                           the Freeze Index included, so the comparison stays like for like.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .stats import bootstrap_mean, bootstrap_metric, paired_comparison, paired_mean_diff, per_subject

ROOT = Path(__file__).resolve().parents[1]
FI = "Freeze Index"
RUNS = [  # (results folder, model in that run, display name)
    ("daphnet", FI, FI),
    ("daphnet", "TCN-causal", "TCN-causal"),
    ("daphnet", "TCN-acausal", "TCN-acausal"),
    ("daphnet", "LSTM", "LSTM"),
    ("daphnet", "BiLSTM", "BiLSTM"),
    ("daphnet_aug", "TCN-causal", "TCN-causal, deeper + noise aug."),
    ("daphnet_spectral_aug", "TCN-causal", "TCN-spectral + noise aug."),
    ("daphnet_spectral", "TCN-causal", "TCN-spectral"),
]
# Causal TCN variants competing for "the" learned model. The winner is chosen by mean VALIDATION
# PR-AUC (best epoch, validation patients only), never by test results.
CANDIDATES = ["TCN-causal", "TCN-causal, deeper + noise aug.", "TCN-spectral + noise aug.", "TCN-spectral"]
MODES = [("online", "threshold"), ("online_pp", "threshold+debounce")]


def load_rows(results_root: Path = ROOT / "results") -> tuple[List[Dict], float, List[str]]:
    rows, rate, shown = [], None, []
    for folder, model, name in RUNS:
        p = results_root / folder / "results.json"
        if not p.exists():
            continue
        res = json.load(open(p))
        rate = res["meta"]["rate_hz"]
        for r in res["rows"]:
            if r["model"] == model and r["mode"] in dict(MODES):
                rows.append({**r, "model": name})
        shown.append(name)
    return rows, rate, shown


def _ci(d, k=1):
    return f"{d['point']:.{k}f} [{d['lo']:.{k}f}, {d['hi']:.{k}f}]"


def summarize(results_root: Path = ROOT / "results", n_boot: int = 2000) -> Dict:
    rows, rate, names = load_rows(results_root)
    if not rows:
        raise SystemExit("no Daphnet results found")
    out = {"rate_hz": rate, "n_boot": n_boot, "detectors": {}, "vs_freeze_index": {}}
    val = {n: float(np.mean([r["val_pr_auc"] for r in rows if r["model"] == n and r["mode"] == "online"]))
           for n in names if n in CANDIDATES}
    out["validation_pr_auc"] = val
    out["selected_by_validation"] = max(val, key=val.get) if val else None
    for name in names:
        for mode, label in MODES:
            if not any(r["model"] == name and r["mode"] == mode for r in rows):
                continue
            auc = per_subject(rows, name, mode, lambda r: r["pr_auc"])
            out["detectors"][f"{name} | {label}"] = {
                "event_f1": bootstrap_metric(rows, name, mode, rate, n_boot=n_boot),
                "event_f1_strict": bootstrap_metric(rows, name, mode, rate, key="counts_strict", n_boot=n_boot)
                if all("counts_strict" in r for r in rows if r["model"] == name and r["mode"] == mode) else None,
                "event_recall": bootstrap_metric(rows, name, mode, rate, "event_recall", n_boot=n_boot),
                "event_precision": bootstrap_metric(rows, name, mode, rate, "event_precision", n_boot=n_boot),
                "false_alarms_per_min": bootstrap_metric(rows, name, mode, rate, "false_alarms_per_min", n_boot=n_boot),
                "pr_auc": bootstrap_mean(auc, n_boot=n_boot),
                "median_debounce_s": float(np.median([r.get("min_dur", 0) for r in rows
                                                      if r["model"] == name and r["mode"] == mode]) / rate),
            }
            if name != FI:
                out["vs_freeze_index"][f"{name} | {label}"] = {
                    "event_f1": paired_comparison(rows, name, FI, rate, mode, mode, n_boot=n_boot),
                    "event_f1_strict": paired_comparison(rows, name, FI, rate, mode, mode, key="counts_strict",
                                                         n_boot=n_boot),
                    "pr_auc": paired_mean_diff(auc, per_subject(rows, FI, mode, lambda r: r["pr_auc"]), n_boot=n_boot),
                }
    return out


def write(results_root: Path = ROOT / "results", n_boot: int = 2000) -> Dict:
    s = summarize(results_root, n_boot)
    L = ["# Daphnet: headline results with uncertainty", "",
         "Online (deployable) setting, 8 test subjects x 3 seeds. Point estimate = pooled over test subjects "
         "within a seed, mean over seeds. **[95% CI]** = percentile bootstrap over test subjects "
         f"({n_boot} resamples of the 8 patients; the patient is the unit of independence). PR-AUC = mean "
         "over subjects of the per-subject value (mean over seeds). *Strict* F1 uses one-to-one event "
         "matching (`event_counts(strict=True)`), a sensitivity check of the event definition. Every "
         "threshold and debounce was chosen on validation subjects only.", "",
         "**Model selection.** Among the causal TCN variants, the one with the highest mean validation PR-AUC "
         "(validation patients only) is selected: **" + str(s["selected_by_validation"]) + "** (mean validation PR-AUC: "
         + ", ".join(f"{k} {v:.3f}" for k, v in s["validation_pr_auc"].items()) + "). Test results played no part "
         "in this choice; the other variants are shown as ablations.", "",
         "| Detector | Operating point | Event F1 [95% CI] | Strict event F1 | Recall | Precision | "
         "False alarms / min | PR-AUC [95% CI] | Debounce (median) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, d in s["detectors"].items():
        name, label = key.split(" | ")
        st = _ci(d["event_f1_strict"]) if d["event_f1_strict"] else "n/a"
        L.append(f"| {name} | {label} | {_ci(d['event_f1'])} | {st} | {d['event_recall']['point']:.1f} | "
                 f"{d['event_precision']['point']:.1f} | {d['false_alarms_per_min']['point']:.2f} | "
                 f"{_ci(d['pr_auc'], 3)} | {d['median_debounce_s']:.2f} s |")
    L += ["", "## Paired comparison with the Freeze Index (same operating-point rule)", "",
          "Difference in event F1 (points). CI = paired bootstrap over test subjects; P(>0) = fraction of "
          "bootstrap resamples in which the learned model is better; Wilcoxon = exact two-sided signed-rank "
          "test on per-subject F1 (smallest attainable p with 8 subjects is 0.008); wins = subjects where "
          "the learned model has the higher F1. Delta PR-AUC = mean over subjects of the per-subject difference "
          "(the Freeze Index scored on its power-gated FI, the signal its detector uses).", "",
          "| Detector | Operating point | Delta F1 [95% CI] | P(>0) | Wilcoxon p | Wins | Delta strict F1 [95% CI] | Delta PR-AUC [95% CI] | PR-AUC Wilcoxon p | PR-AUC wins |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, d in s["vs_freeze_index"].items():
        name, label = key.split(" | ")
        e, es = d["event_f1"], d["event_f1_strict"]
        L.append(f"| {name} | {label} | {e['diff']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}] | {e['p_boot_gt0']:.2f} | "
                 f"{e['wilcoxon_p']:.3f} | {e['wins']}/{e['n_subjects']} | "
                 f"{es['diff']:+.1f} [{es['lo']:+.1f}, {es['hi']:+.1f}] | "
                 f"{d['pr_auc']['diff']:+.3f} [{d['pr_auc']['lo']:+.3f}, {d['pr_auc']['hi']:+.3f}] | "
                 f"{d['pr_auc']['wilcoxon_p']:.3f} | {d['pr_auc']['wins']}/{d['pr_auc']['n_subjects']} |")
    L += ["", "Read with care: 8 patients from one lab study. These are descriptive statistics, not a "
          "confirmatory test, and nothing here is a clinical claim."]
    (results_root / "daphnet_summary.md").write_text("\n".join(L) + "\n")
    json.dump(s, open(results_root / "daphnet_summary.json", "w"), indent=1)
    _figure(s, results_root / "daphnet_summary.png")
    print("\n".join(L))
    return s


def _figure(s: Dict, path: Path) -> None:
    items = [(k, d) for k, d in s["detectors"].items()]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(items) + 1.2))
    for i, (k, d) in enumerate(items):
        e = d["event_f1"]
        color = "#7f7f7f" if k.startswith(FI) else ("#1f77b4" if "+" in k.split(" | ")[0] else "#9ecae1")
        ax.errorbar(e["point"], i, xerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="o",
                    color=color, capsize=3)
    fi = [d["event_f1"]["point"] for k, d in items if k.startswith(FI)]
    for v in fi:
        ax.axvline(v, color="#7f7f7f", ls=":", lw=1)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([k.replace(" | ", ", ") for k, _ in items], fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("Online event F1 (%), 95% CI over test subjects")
    ax.set_title("Daphnet freezing-of-gait detection (8 held-out patients)")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


if __name__ == "__main__":
    write()
