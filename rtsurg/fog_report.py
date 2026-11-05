"""Tables and figures for the Daphnet benchmark (everything is computed from ``results.json``)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .fog import summarize_counts
from .fog_run import FI_NAME
from .fog import merge_counts

COLORS = {"TCN-causal": "#1f77b4", "LSTM": "#2ca02c", "TCN-acausal": "#ff7f0e",
          "BiLSTM": "#d62728", FI_NAME: "#7f7f7f"}


def _ms(v, d=1):
    v = np.asarray([x for x in v if x == x], float)
    return "n/a" if not len(v) else f"{v.mean():.{d}f} ± {v.std():.{d}f}"


def aggregate(rows: List[Dict], rate: float, model: str, mode: str) -> Dict:
    r = [x for x in rows if x["model"] == model and x["mode"] == mode]
    seeds = sorted({x["seed"] for x in r})
    pooled = [summarize_counts(merge_counts([x["counts"] for x in r if x["seed"] == s]), rate)
              for s in seeds]
    allc = merge_counts([x["counts"] for x in r])
    return {
        "pooled_by_seed": pooled,
        "pooled_all": summarize_counts(allc, rate),
        "fold_summ": [x["summary"] for x in r],
        "pr_auc": [x["pr_auc"] for x in r],
        "by_fold": {f: [x for x in r if x["fold"] == f] for f in sorted({x["fold"] for x in r})},
        "thr": [x["thr"] for x in r],
    }


def write_daphnet_report(res: Dict, out: Path) -> None:
    import json
    res = json.loads(json.dumps(res, default=float))      # same view whether live or loaded from disk
    out = Path(out); (out / "figures").mkdir(parents=True, exist_ok=True)
    rows, meta, lat = res["rows"], res["meta"], res["latency"]
    rate, models = meta["rate_hz"], res["models"]
    names = [FI_NAME] + models
    causal = {"TCN-causal": True, "LSTM": True, "TCN-acausal": False, "BiLSTM": False, FI_NAME: True}
    ref_frames = str(int(60 * rate))
    agg = {(m, mode): aggregate(rows, rate, m, mode) for m in names for mode in ("offline", "online", "online_pp")
           if any(x["model"] == m and x["mode"] == mode for x in rows)}
    folds = sorted({x["fold"] for x in rows})
    L = []
    L += [f"# Daphnet Freezing of Gait: leave-one-subject-out results", "",
          f"Seeds {meta['seeds']} · {len(folds)} test subjects with freezes ({', '.join(folds)}) · "
          f"{rate:.0f} Hz · frame budget {meta['frame_ms']:.1f} ms · {meta['epochs']} epochs · "
          f"hidden {meta['hidden']} · TCN layers {meta['tcn_layers']} · {meta['n_val']} validation subjects per fold · "
          f"input features: {meta.get('features', 'raw')} · training noise augmentation: "
          f"{'up to ' + str(meta['augment_noise']) + 'x feature std' if meta.get('augment_noise') else 'none'}", "",
          "All thresholds were selected on validation subjects only. Event metrics are pooled over all "
          "test subjects within a seed (mean ± std across seeds). Sensitivity, specificity and PR-AUC are "
          "per fold (mean ± std across seeds and folds).", "",
          "## Main table: online (deployable) setting", "",
          "| Model | Causal | Event precision | Event recall | Event F1 | Sensitivity | Specificity | PR-AUC | Median onset delay (ms) | False alarms / min | Latency median / p95 (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in names:
        a = agg[(m, "online")]
        ps = a["pooled_by_seed"]
        lt = lat.get(m, {}).get("by_history", {}).get(ref_frames)
        lat_s = f"{lt['median_ms']:.2f} / {lt['p95_ms']:.2f}" if lt else "n/a (FFT, no network)"
        L.append(f"| {m} | {'yes' if causal[m] else 'no'} | {_ms([p['event_precision'] for p in ps])} | "
                 f"{_ms([p['event_recall'] for p in ps])} | {_ms([p['event_f1'] for p in ps])} | "
                 f"{_ms([f['sensitivity'] for f in a['fold_summ']])} | {_ms([f['specificity'] for f in a['fold_summ']])} | "
                 f"{_ms(a['pr_auc'], 3)} | {a['pooled_all']['median_delay_ms']:.0f} | "
                 f"{_ms([p['false_alarms_per_min'] for p in ps], 2)} | {lat_s} |")
    if any((m, "online_pp") in agg for m in names):
        L += ["", "## Online with a debounced alarm (threshold and minimum alarm duration both chosen on validation subjects)", "",
              "An alarm is raised once p(freeze) has stayed above the threshold for the chosen duration (0 to 1.5 s); "
              "the same rule is applied to the Freeze Index. PR-AUC of the Freeze Index is computed on the "
              "power-gated FI score that its detector uses.", "",
              "| Model | Event precision | Event recall | Event F1 | False alarms / min | Median onset delay (ms) | Median debounce (s) |",
              "|---|---|---|---|---|---|---|"]
        for m in names:
            if (m, "online_pp") not in agg:
                continue
            a = agg[(m, "online_pp")]; ps = a["pooled_by_seed"]
            md = np.median([x.get("min_dur", 0) for x in rows if x["model"] == m and x["mode"] == "online_pp"]) / rate
            L.append(f"| {m} | {_ms([p['event_precision'] for p in ps])} | {_ms([p['event_recall'] for p in ps])} | "
                     f"{_ms([p['event_f1'] for p in ps])} | {_ms([p['false_alarms_per_min'] for p in ps], 2)} | "
                     f"{a['pooled_all']['median_delay_ms']:.0f} | {md:.2f} |")
    L += ["", "## Offline vs. online (event F1, pooled; mean ± std across seeds)", "",
          "| Model | Offline event F1 | Online event F1 | Change | Offline PR-AUC | Online PR-AUC |", "|---|---|---|---|---|---|"]
    for m in models:
        off, on = agg[(m, "offline")], agg[(m, "online")]
        fo = np.mean([p["event_f1"] for p in off["pooled_by_seed"]])
        fn = np.mean([p["event_f1"] for p in on["pooled_by_seed"]])
        L.append(f"| {m} | {_ms([p['event_f1'] for p in off['pooled_by_seed']])} | "
                 f"{_ms([p['event_f1'] for p in on['pooled_by_seed']])} | {fn - fo:+.1f} | "
                 f"{_ms(off['pr_auc'], 3)} | {_ms(on['pr_auc'], 3)} |")
    L += ["", "## Per-subject online event F1 (mean over seeds)", "",
          "| Model | " + " | ".join(folds) + " |", "|---|" + "---|" * len(folds)]
    for m in names:
        a = agg[(m, "online")]
        cells = []
        for f in folds:
            v = [x["summary"]["event_f1"] for x in a["by_fold"][f]]
            cells.append("n/a" if not v or np.isnan(v).all() else f"{np.nanmean(v):.1f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "## Per-subject online recall / false alarms per minute (mean over seeds)", "",
          "| Model | " + " | ".join(folds) + " |", "|---|" + "---|" * len(folds)]
    for m in names:
        a = agg[(m, "online")]
        cells = [f"{np.nanmean([x['summary']['event_recall'] for x in a['by_fold'][f]]):.0f}% / "
                 f"{np.mean([x['summary']['false_alarms_per_min'] for x in a['by_fold'][f]]):.1f}" for f in folds]
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    ref = agg[(FI_NAME, "online")]
    L += ["", "## Class balance of the test subjects", "",
          "| Subject | Freeze frames (%) | Freeze episodes | Test minutes |", "|---|---|---|---|"]
    for f in folds:
        x = ref["by_fold"][f][0]
        c = x["counts"]
        L.append(f"| {f} | {x['summary']['freeze_frac']:.1f} | {c['n_true']} | {c['frames'] / rate / 60:.1f} |")
    allc = merge_counts([ref["by_fold"][f][0]["counts"] for f in folds])
    L += [f"| **All** | {100 * (allc['tp'] + allc['fn']) / allc['frames']:.1f} | {allc['n_true']} | "
          f"{allc['frames'] / rate / 60:.1f} |", "",
          "## Per-update latency (CPU, batch size 1, median / p95 ms) by elapsed history", "",
          "| Model | " + " | ".join(f"{s} s" for s in meta["hist_s"]) + " |", "|---|" + "---|" * len(meta["hist_s"])]
    for m in models:
        cells = []
        for s in meta["hist_s"]:
            v = lat[m]["by_history"][str(int(s * rate))]
            cells.append(f"{v['median_ms']:.2f} / {v['p95_ms']:.2f}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", f"Latency measured on `{meta['processor'] or meta['platform']}`; online acausal models re-run on the "
          f"last {meta['online_history_s']:.0f} s every {meta['recompute_stride']} frames "
          f"({meta['recompute_stride'] * meta['frame_ms']:.0f} ms), as a deployed system would have to."]
    (out / "results.md").write_text("\n".join(L) + "\n")

    # ---- figures ---------------------------------------------------------------------
    fig = out / "figures"
    # 1. offline vs online event F1
    fig1, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(models)); w = 0.38
    for i, mode in enumerate(("offline", "online")):
        vals = [[p["event_f1"] for p in agg[(m, mode)]["pooled_by_seed"]] for m in models]
        ax.bar(x + (i - 0.5) * w, [np.mean(v) for v in vals], w, yerr=[np.std(v) for v in vals],
               label=mode, color=["#4c78a8", "#f58518"][i], capsize=3)
    fi_f1 = np.mean([p["event_f1"] for p in ref["pooled_by_seed"]])
    ax.axhline(fi_f1, color=COLORS[FI_NAME], ls="--", label="Freeze Index baseline")
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Event F1 (%), pooled")
    ax.set_title("Daphnet: offline vs. online"); ax.legend(frameon=False); fig1.tight_layout()
    fig1.savefig(fig / "offline_vs_online.png", dpi=160); plt.close(fig1)
    # 2. per-subject F1
    fig2, ax = plt.subplots(figsize=(8, 3.8))
    k = len(names); w = 0.8 / k
    for i, m in enumerate(names):
        v = [np.nanmean([y["summary"]["event_f1"] for y in agg[(m, "online")]["by_fold"][f]]) for f in folds]
        ax.bar(np.arange(len(folds)) + (i - (k - 1) / 2) * w, v, w, label=m, color=COLORS[m])
    ax.set_xticks(range(len(folds))); ax.set_xticklabels(folds)
    ax.set_ylabel("Online event F1 (%)"); ax.set_title("Per-subject online event F1 (mean over seeds)")
    ax.legend(frameon=False, ncol=3, fontsize=8); fig2.tight_layout()
    fig2.savefig(fig / "per_subject_f1.png", dpi=160); plt.close(fig2)
    # 3. latency vs history
    fig3, ax = plt.subplots(figsize=(6, 3.8))
    for m in models:
        ys = [lat[m]["by_history"][str(int(s * rate))]["median_ms"] for s in meta["hist_s"]]
        ax.plot(meta["hist_s"], ys, marker="o", label=m, color=COLORS[m])
    ax.axhline(meta["frame_ms"], color="k", ls=":", label=f"frame budget {meta['frame_ms']:.0f} ms")
    ax.set_yscale("log"); ax.set_xlabel("Elapsed recording time (s)"); ax.set_ylabel("Median per-update latency (ms)")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Latency vs. elapsed time (CPU)"); fig3.tight_layout()
    fig3.savefig(fig / "latency_vs_history.png", dpi=160); plt.close(fig3)
    # 4. event recall vs false alarms
    fig4, ax = plt.subplots(figsize=(5.5, 4))
    for m in names:
        ps = agg[(m, "online")]["pooled_by_seed"]
        ax.errorbar(np.mean([p["false_alarms_per_min"] for p in ps]), np.mean([p["event_recall"] for p in ps]),
                    xerr=np.std([p["false_alarms_per_min"] for p in ps]),
                    yerr=np.std([p["event_recall"] for p in ps]), fmt="o", label=m, color=COLORS[m], capsize=3)
    ax.set_xlabel("False alarms per minute"); ax.set_ylabel("Episode recall (%)")
    ax.set_title("Online operating points"); ax.legend(frameon=False, fontsize=8); fig4.tight_layout()
    fig4.savefig(fig / "recall_vs_false_alarms.png", dpi=160); plt.close(fig4)
    print(f"[done] results -> {out / 'results.md'}, figures -> {fig}")
