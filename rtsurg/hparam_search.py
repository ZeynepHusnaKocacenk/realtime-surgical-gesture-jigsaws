"""Validation-only hyperparameter search for the causal TCN on Daphnet.

    python -m rtsurg.hparam_search --data-dir data/daphnet --workers 4

For every configuration and every leave-one-subject-out fold (seed 0, the same splits as the
benchmark), the model is trained on the training patients and scored on that fold's VALIDATION
patients only: best-epoch PR-AUC, and event F1 at the threshold that is best on those same patients.
The test patient of each fold is never featurized, predicted or scored here, so choosing the
configuration with the highest mean validation PR-AUC cannot leak test information. The chosen
configuration is then run on the full 3-seed protocol (``configs/daphnet_tuned.yaml``).

Validation scores are optimistic (epoch and threshold are also picked on them), equally for every
configuration, so they rank configurations but are not performance estimates.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .daphnet import SAMPLING_HZ, load_daphnet, loso_splits
from .data import Normalizer
from .features import featurize
from .fog import event_counts, freeze_prob, predict_logits, select_threshold, summarize_counts, train_fog_model
from .models import build_model

ROOT = Path(__file__).resolve().parents[1]
BASE = dict(features="spectral", hidden=32, layers=8, lr=2e-3, dropout=0.3, epochs=20, window=256)


def grid() -> List[Dict]:
    """Architecture grid (hidden x layers) plus one-at-a-time variations around the centre."""
    cfgs = [dict(BASE, hidden=h, layers=l) for h, l in itertools.product([16, 32, 64], [6, 8, 10])]
    cfgs += [dict(BASE, lr=v) for v in (1e-3, 5e-3)]
    cfgs += [dict(BASE, dropout=v) for v in (0.1, 0.5)]
    cfgs += [dict(BASE, window=v) for v in (128, 512)]
    cfgs += [dict(BASE, features="raw")]
    return cfgs


def name(c: Dict) -> str:
    return (f"{c['features']}_h{c['hidden']}_L{c['layers']}_lr{c['lr']:g}_do{c['dropout']:g}_w{c['window']}")


def run_config(cfg: Dict, data_dir: str, seed: int = 0) -> Dict:
    torch.set_num_threads(1)
    data = load_daphnet(data_dir, downsample=2)
    rate = SAMPLING_HZ / 2
    fz = lambda ts: [dataclasses.replace(t, kinematics=featurize(t.kinematics, cfg["features"], rate)) for t in ts]
    folds, t0 = [], time.time()
    for user, train, val, _test in loso_splits(data, 2, seed):      # _test is deliberately never touched
        train_f, val_f = fz(train), fz(val)
        norm = Normalizer.fit(train_f)
        torch.manual_seed(seed)
        model = build_model("TCN-causal", train_f[0].kinematics.shape[1], 2, cfg["hidden"], cfg["layers"],
                            dropout=cfg["dropout"])
        _, ap = train_fog_model(model, train_f, val_f, norm, rate, cfg["epochs"], seed,
                                win=cfg["window"], stride=cfg["window"], lr=cfg["lr"], eval_every=2)
        vx = [norm(t.kinematics) for t in val_f]; vy = [t.gestures for t in val]
        p = [freeze_prob(l) for l in predict_logits(model, vx)]
        thr = select_threshold(p, vy, rate)
        f1 = summarize_counts(event_counts([q >= thr for q in p], vy, rate), rate)["event_f1"]
        folds.append({"fold": user, "val_users": sorted({t.user for t in val}), "val_pr_auc": ap, "val_event_f1": f1})
    params = sum(q.numel() for q in model.parameters())
    out = {"name": name(cfg), "cfg": cfg, "params": params, "folds": folds, "minutes": (time.time() - t0) / 60,
           "val_pr_auc": float(np.mean([f["val_pr_auc"] for f in folds])),
           "val_event_f1": float(np.mean([f["val_event_f1"] for f in folds]))}
    print(f"[search] {out['name']:42s} val PR-AUC {out['val_pr_auc']:.3f}  val F1 {out['val_event_f1']:5.1f}  "
          f"({params:,} params, {out['minutes']:.0f} min)", flush=True)
    return out


def report(res: List[Dict], out: Path) -> Dict:
    res = sorted(res, key=lambda r: -r["val_pr_auc"])
    best = res[0]
    L = ["# Validation-only hyperparameter search (causal TCN, Daphnet)", "",
         f"{len(res)} configurations x 8 leave-one-subject-out folds (seed 0). Each configuration is scored on the "
         "**validation patients** of each fold only; the test patients were never used. Ranked by mean validation "
         "PR-AUC (the selection criterion); validation event F1 is shown for reference. Both are optimistic "
         "(epoch and threshold are also chosen on the validation patients) and are not performance estimates.", "",
         f"**Selected:** `{best['name']}` (mean validation PR-AUC {best['val_pr_auc']:.3f}). It is then evaluated on the "
         "test patients with 3 seeds (`configs/daphnet_tuned.yaml`).", "",
         "| Rank | Features | Hidden | Layers | Receptive field (s) | LR | Dropout | Window (s) | Params | Val PR-AUC | Val event F1 | CPU min |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(res, 1):
        c = r["cfg"]; rf = (1 + 2 * (2 ** c["layers"] - 1)) / 32.0
        L.append(f"| {i} | {c['features']} | {c['hidden']} | {c['layers']} | {rf:.1f} | {c['lr']:g} | {c['dropout']:g} | "
                 f"{c['window'] / 32:.0f} | {r['params']:,} | {r['val_pr_auc']:.3f} | {r['val_event_f1']:.1f} | {r['minutes']:.0f} |")
    (out / "search.md").write_text("\n".join(L) + "\n")
    json.dump({"results": res, "selected": best["name"], "selected_cfg": best["cfg"]}, open(out / "search.json", "w"), indent=1)
    _figures(res, out)
    return best


def _figures(res: List[Dict], out: Path) -> None:
    by = {name(r["cfg"]): r for r in res}
    hs, ls = [16, 32, 64], [6, 8, 10]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    M = np.full((len(hs), len(ls)), np.nan)
    for i, h in enumerate(hs):
        for j, l in enumerate(ls):
            r = by.get(name(dict(BASE, hidden=h, layers=l)))
            if r:
                M[i, j] = r["val_pr_auc"]
    ax = axes[0]
    im = ax.imshow(M, cmap="Blues", vmin=np.nanmin(M) - 0.02, vmax=np.nanmax(M) + 0.01)
    for i in range(len(hs)):
        for j in range(len(ls)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if M[i, j] > np.nanmean(M) else "black")
    ax.set_xticks(range(len(ls))); ax.set_xticklabels([f"{l}\n({(1 + 2 * (2 ** l - 1)) / 32:.0f} s)" for l in ls])
    ax.set_yticks(range(len(hs))); ax.set_yticklabels(hs)
    ax.set_xlabel("TCN layers (receptive field)"); ax.set_ylabel("hidden channels")
    ax.set_title("Mean validation PR-AUC (spectral input)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax = axes[1]
    rs = sorted(res, key=lambda r: r["val_pr_auc"])
    for k, r in enumerate(rs):                     # dot plot: an axis not starting at 0 is honest for dots
        sel = r is res[0]
        ax.plot([lo_x := min(q["val_pr_auc"] for q in rs) - 0.05, r["val_pr_auc"]], [k, k], color="#e4e3df", lw=1, zorder=1)
        ax.scatter(r["val_pr_auc"], k, s=55 if sel else 35, color="#1f5fbf" if sel else "#7fa7d9", zorder=2,
                   edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(rs))); ax.set_yticklabels([r["name"].replace("_", " ") for r in rs], fontsize=6.5)
    lo = min(r["val_pr_auc"] for r in rs)
    ax.set_xlim(lo - 0.05, max(r["val_pr_auc"] for r in rs) + 0.02)
    ax.set_xlabel("mean validation PR-AUC (8 folds)"); ax.set_title("All 16 configurations (selected: largest dot)", fontsize=10)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    fig.tight_layout(); fig.savefig(out / "search.png", dpi=160); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "daphnet"))
    ap.add_argument("--out", default=str(ROOT / "results" / "hparam_search"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    part = out / "partial.jsonl"
    done = {json.loads(l)["name"]: json.loads(l) for l in part.read_text().splitlines()} if part.exists() else {}
    todo = [c for c in grid() if name(c) not in done]
    if not args.report_only and todo:
        todo.sort(key=lambda c: -c["hidden"] * c["layers"])        # longest first for better packing
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(args.workers) as ex:
            for fut in as_completed([ex.submit(run_config, c, args.data_dir) for c in todo]):
                r = fut.result()
                with open(part, "a") as f:
                    f.write(json.dumps(r) + "\n")
                done[r["name"]] = r
    best = report(list(done.values()), out)
    print(f"[search] selected {best['name']}")


if __name__ == "__main__":
    main()
