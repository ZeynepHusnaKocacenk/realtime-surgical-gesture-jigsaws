"""Configs must be valid, and the report must never show synthetic numbers as JIGSAWS results."""
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_all_configs_use_known_settings():
    sys.path.insert(0, str(ROOT))
    import main as m
    dests = set()
    old = sys.argv
    try:
        sys.argv = ["main.py", "--synthetic"]
        # build the parser through parse_args to collect every dest
        import argparse
        orig = argparse.ArgumentParser.set_defaults
        seen = {}
        def spy(self, **kw):
            seen.update(kw); return orig(self, **kw)
        for cfg in sorted((ROOT / "configs").glob("*.yaml")):
            argparse.ArgumentParser.set_defaults = spy
            sys.argv = ["main.py", "--config", str(cfg)]
            args = m.parse_args() if yaml.safe_load(open(cfg)).get("data_dir") else None
            assert args is not None and args.dataset in ("daphnet", "jigsaws")
            assert list(args.seeds) == [0, 1, 2]
    finally:
        argparse.ArgumentParser.set_defaults = orig
        sys.argv = old


def test_daphnet_config_matches_saved_results():
    from rtsurg.config import daphnet_results_match
    import argparse
    cfg = yaml.safe_load(open(ROOT / "configs" / "daphnet.yaml"))
    saved = ROOT / "results" / "daphnet" / "results.json"
    if not saved.exists():
        return
    a = argparse.Namespace(seeds=cfg["seeds"], epochs=cfg["daphnet_epochs"], hidden=cfg["daphnet_hidden"],
                           tcn_layers=cfg["daphnet_tcn_layers"], n_val=cfg["n_val"], window=cfg["window"],
                           lr=cfg["daphnet_lr"], online_history_s=cfg["online_history_s"],
                           recompute_stride=cfg["recompute_stride"], downsample=cfg["daphnet_downsample"],
                           models=cfg["models"])
    assert daphnet_results_match(a, saved)
    a.seeds = [0, 1]
    assert not daphnet_results_match(a, saved)


def test_report_hides_synthetic_jigsaws(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT))
    import make_report as mr
    monkeypatch.setattr(mr, "RES", tmp_path / "results"); monkeypatch.setattr(mr, "DOCS", tmp_path / "docs")
    d = tmp_path / "results" / "jigsaws_suturing"; d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({"meta": {"synthetic": True}}))
    (d / "results.md").write_text("# Results\n\n| a | b |\n|---|---|\n| 99.9 | SYNTHETIC |\n")
    mr.main()
    page = (tmp_path / "docs" / "index.html").read_text()
    assert "Awaiting dataset access" in page and "SYNTHETIC" not in page and "99.9" not in page
    assert "JIGSAWS results pending dataset access; real results currently on Daphnet" in page
