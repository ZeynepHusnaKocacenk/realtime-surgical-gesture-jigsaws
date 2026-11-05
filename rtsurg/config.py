"""Check whether saved results were produced by the settings in a config (so they need no re-run)."""
from __future__ import annotations

import json
from pathlib import Path


def _load_meta(path: Path):
    if not Path(path).exists():
        return None
    try:
        return json.load(open(path))
    except Exception:
        return None


def daphnet_results_match(args, path: Path) -> bool:
    res = _load_meta(path)
    if not res:
        return False
    m = res["meta"]
    want = {"seeds": list(args.seeds), "epochs": args.epochs, "hidden": args.hidden,
            "tcn_layers": args.tcn_layers, "n_val": args.n_val, "window": args.window,
            "lr": args.lr, "online_history_s": args.online_history_s,
            "recompute_stride": args.recompute_stride, "downsample": args.downsample,
            "features": getattr(args, "features", "raw"),
            "augment_noise": float(getattr(args, "augment_noise", 0.0) or 0.0)}
    m = {**{"features": "raw", "augment_noise": 0.0}, **m}       # results saved before these options
    return all(m.get(k) == v for k, v in want.items()) and list(res.get("models", [])) == list(args.models)


def robustness_results_match(args, path: Path) -> bool:
    res = _load_meta(path)
    if not res:
        return False
    m = res["meta"]
    return m.get("seeds") == list(args.seeds) and m.get("models") == list(args.models)
