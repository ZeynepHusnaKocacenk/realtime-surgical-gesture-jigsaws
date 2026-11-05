"""Find and validate JIGSAWS task folders before spending hours on training.

JIGSAWS unzips to one folder per task (Suturing, Knot_Tying, Needle_Passing), each holding
``kinematics/AllGestures``, ``transcriptions`` and ``meta_file_<Task>.txt``. Some archives nest the
folder one level deeper (``Suturing/Suturing/...``); ``find_task_dir`` handles both.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .data import N_KIN_COLS, TRIAL_RE, _parse_meta, _parse_transcription

TASKS = ["Suturing", "Knot_Tying", "Needle_Passing"]


def find_task_dir(root: str | Path, task: str) -> Optional[Path]:
    """Return the folder that directly contains ``meta_file_<task>.txt`` below ``root`` (or None)."""
    root = Path(root)
    for meta in [root / f"meta_file_{task}.txt", *sorted(root.rglob(f"meta_file_{task}.txt"))]:
        if meta.exists() and (meta.parent / "kinematics" / "AllGestures").is_dir():
            return meta.parent
    return None


def check_task_dir(task_dir: str | Path, min_users: int = 4) -> List[str]:
    """Return a list of problems (empty list = looks usable). Reads only headers and line counts."""
    d, problems = Path(task_dir), []
    task = d.name
    metas = list(d.glob("meta_file_*.txt"))
    if not metas:
        return [f"{d}: no meta_file_*.txt"]
    meta = _parse_meta(metas[0])
    if not meta:
        problems.append(f"{metas[0].name}: no trial lines with a skill letter (N/I/E) were parsed")
    kin_dir, tr_dir = d / "kinematics" / "AllGestures", d / "transcriptions"
    for sub in (kin_dir, tr_dir):
        if not sub.is_dir():
            problems.append(f"missing folder {sub}")
    if problems:
        return problems
    usable, users = 0, set()
    for tr in sorted(tr_dir.glob("*.txt")):
        m = TRIAL_RE.match(tr.stem)
        kin = kin_dir / tr.name
        if m is None:
            problems.append(f"{tr.name}: name does not look like <Task>_<User><3 digits>"); continue
        if tr.stem not in meta:
            problems.append(f"{tr.stem}: not listed in the meta file (will be skipped)"); continue
        if not kin.exists():
            problems.append(f"{tr.stem}: no kinematics file (will be skipped)"); continue
        first = kin.read_text().splitlines()[0].split() if kin.stat().st_size else []
        if len(first) != N_KIN_COLS:
            problems.append(f"{kin.name}: {len(first)} columns, expected {N_KIN_COLS}"); continue
        n = sum(1 for _ in open(kin))
        segs = _parse_transcription(tr)
        if not segs or max(e for _, e, _ in segs) > n + 1:
            problems.append(f"{tr.name}: transcription missing or runs past the {n} kinematic frames"); continue
        usable += 1; users.add(m.group("user"))
    if usable == 0:
        problems.append(f"{task}: no usable trials")
    elif len(users) < min_users:
        problems.append(f"{task}: only {len(users)} surgeons found ({sorted(users)}), leave-one-user-out needs more")
    return problems


def summarize(task_dir: str | Path) -> str:
    from .data import load_jigsaws_task
    data = load_jigsaws_task(task_dir)
    skills = {u: next(t.skill for t in data.trials if t.user == u) for u in data.users}
    frames = sum(len(t.gestures) for t in data.trials)
    return (f"{Path(task_dir).name}: {len(data.trials)} trials, surgeons {data.users}, "
            f"{data.n_gestures} gesture classes, {frames} labelled frames, skill per surgeon {skills}")
