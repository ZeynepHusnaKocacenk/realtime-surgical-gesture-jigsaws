"""Run everything for JIGSAWS with one command (standard library only).

    python a.py /path/to/JIGSAWS

The folder must contain Suturing/, Knot_Tying/ and Needle_Passing/ (the unzipped dataset).
Steps: get the code, install requirements, validate the data, quick sanity run, full benchmark,
then (after asking) commit and push the results so the project page updates.
Options: --yes (do not ask), --no-push, --skip-quick, --tasks Suturing ...
Run it from anywhere; if it is not inside the repository it clones it next to itself.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/ZeynepHusnaKocacenk/realtime-surgical-gesture-jigsaws"


def run(cmd, cwd, check=True):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    rc = subprocess.call([str(c) for c in cmd], cwd=cwd)
    if check and rc != 0:
        sys.exit(f"\nStopped: the command above failed (exit code {rc}). Fix it and run a.py again.")
    return rc


def ask(question, assume_yes):
    if assume_yes:
        return True
    try:
        return input(f"\n{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", help="folder containing the JIGSAWS task folders")
    ap.add_argument("--tasks", nargs="+", default=None, help="e.g. Suturing (default: all three)")
    ap.add_argument("--yes", action="store_true", help="do not ask before the long run and the push")
    ap.add_argument("--no-push", action="store_true", help="do not commit or push the results")
    ap.add_argument("--skip-quick", action="store_true", help="skip the quick sanity run")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    root = args.root or input("Path to the JIGSAWS folder (contains Suturing, Knot_Tying, Needle_Passing): ").strip()
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Folder not found: {root}")

    here = Path(__file__).resolve().parent
    if (here / "run_jigsaws.py").exists():
        repo = here
        if (repo / ".git").exists():
            run(["git", "pull", "--ff-only"], repo, check=False)
    else:
        repo = here / "realtime-surgical-gesture-jigsaws"
        if not repo.exists():
            run(["git", "clone", REPO_URL, repo], here)
    if repo in root.parents or root == repo:
        sys.exit("Keep the JIGSAWS data outside the repository folder (its license forbids publishing it).")

    if not args.skip_install:
        run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], repo)

    task_args = ["--tasks", *args.tasks] if args.tasks else []
    base = [sys.executable, "run_jigsaws.py", "--root", root]
    run(base + ["--check"] + task_args, repo)
    if not args.skip_quick:
        run(base + ["--quick"] + task_args, repo)
        print("\nQuick run finished: the pipeline works on your files. (Quick numbers are not a result.)")
    if not ask("Start the FULL benchmark now? It takes several hours on a CPU.", args.yes):
        sys.exit("Stopped before the full run. Run a.py again when you are ready.")
    run(base + task_args, repo)

    print("\nResults: results/jigsaws_<task>/results.md and figures/, project page: docs/index.html")
    if args.no_push:
        return
    if not ask("Commit and push the results and the rebuilt project page to GitHub?", args.yes):
        return
    run(["git", "add", "results", "docs"], repo)
    if subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=repo) == 0:
        print("Nothing new to commit."); return
    run(["git", "commit", "-m", "Add JIGSAWS results"], repo)
    run(["git", "push", "origin", "HEAD"], repo)
    print("\nPushed. The GitHub Pages workflow now rebuilds your project page.")


if __name__ == "__main__":
    main()
