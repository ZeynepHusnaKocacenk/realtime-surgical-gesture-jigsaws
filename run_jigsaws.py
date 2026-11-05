"""One command to run the whole benchmark on JIGSAWS once you have the data.

    python run_jigsaws.py --root /path/to/JIGSAWS            # all three tasks, 3 seeds, all four models
    python run_jigsaws.py --root /path/to/JIGSAWS --tasks Suturing
    python run_jigsaws.py --root /path/to/JIGSAWS --quick     # 1 seed, 2 models, few epochs: sanity run
    python run_jigsaws.py --root /path/to/JIGSAWS --check     # only validate the folders

``--root`` is the folder that contains Suturing/, Knot_Tying/ and Needle_Passing/ (nested archives such
as Suturing/Suturing/ are found automatically). Results go to results/jigsaws_<task>/ and the project page
docs/index.html is rebuilt. JIGSAWS is never copied into the repository.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from rtsurg.jigsaws_check import TASKS, check_task_dir, find_task_dir, summarize

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder containing the JIGSAWS task folders")
    ap.add_argument("--tasks", nargs="+", default=TASKS, choices=TASKS)
    ap.add_argument("--check", action="store_true", help="only validate the data, do not train")
    ap.add_argument("--quick", action="store_true", help="1 seed, TCN-causal + LSTM, 5 epochs, 2 folds (not a result)")
    ap.add_argument("--out-root", default=str(ROOT / "results"), help="where results/jigsaws_<task> are written")
    ap.add_argument("--no-report", action="store_true", help="do not rebuild docs/index.html")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra flags passed to main.py")
    args = ap.parse_args(argv)

    found = {}
    for task in args.tasks:
        d = find_task_dir(args.root, task)
        if d is None:
            print(f"[{task}] not found under {args.root} (looking for meta_file_{task}.txt next to "
                  f"kinematics/AllGestures): skipped")
            continue
        problems = check_task_dir(d)
        fatal = [p for p in problems if "will be skipped" not in p]
        print(f"[{task}] {summarize(d) if not fatal else 'PROBLEMS'}")
        for p in problems:
            print(f"    - {p}")
        if not fatal:
            found[task] = d
    if not found:
        print("No usable JIGSAWS task folder found. Nothing to run."); return 2
    if args.check:
        return 0

    for task, d in found.items():
        cmd = [sys.executable, str(ROOT / "main.py"), "--config", str(ROOT / "configs" / f"jigsaws_{task.lower()}.yaml"),
               "--data-dir", str(d), "--out", str(Path(args.out_root) / f"jigsaws_{task.lower()}")]
        if args.quick:
            cmd += ["--seeds", "0", "--models", "TCN-causal", "LSTM", "--epochs", "5", "--max-folds", "2"]
            cmd[cmd.index("--out") + 1] += "_quick"
        cmd += args.extra
        print(f"\n== {task}: {' '.join(cmd)}", flush=True)
        if subprocess.call(cmd, cwd=ROOT) != 0:
            print(f"[{task}] failed"); return 1
    if not args.no_report and not args.quick:
        subprocess.call([sys.executable, str(ROOT / "make_report.py")], cwd=ROOT)
    print("\nDone. Tables: results/jigsaws_<task>/results.md, figures: results/jigsaws_<task>/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
