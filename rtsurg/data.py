"""JIGSAWS data loading, synthetic data generation and LOUO splitting.

Real JIGSAWS layout (one folder per task, e.g. ``Suturing/``)::

    Suturing/
        kinematics/AllGestures/Suturing_B001.txt   # 76 whitespace-separated columns, 30 Hz
        transcriptions/Suturing_B001.txt           # "<start> <end> G<k>" per line (1-indexed, inclusive)
        meta_file_Suturing.txt                     # "<trial> <N|I|E> <GRS> <6 sub-scores>"

The 76 kinematic columns are 4 manipulators x 19 variables
(MTM-left, MTM-right, PSM-left, PSM-right). Each 19-block is:
position (3), rotation matrix (9), linear velocity (3), angular velocity (3), gripper angle (1).

The synthetic generator writes files in exactly this layout, so the same loader is exercised
on mock data and on the real dataset.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

FRAME_RATE_HZ = 30.0
N_KIN_COLS = 76
BLOCK = 19
MANIPULATORS = {"MTM_L": 0, "MTM_R": 1, "PSM_L": 2, "PSM_R": 3}
SKILL_TO_IDX = {"N": 0, "I": 1, "E": 2}
IDX_TO_SKILL = {v: k for k, v in SKILL_TO_IDX.items()}
TRIAL_RE = re.compile(r"^(?P<task>[A-Za-z_]+)_(?P<user>[A-Z])(?P<rep>\d{3})$")


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------
@dataclass
class Trial:
    name: str                 # e.g. "Suturing_B001"
    task: str
    user: str                 # surgeon id letter, e.g. "B"
    kinematics: np.ndarray    # [T, F] float32, only labelled frames kept
    gestures: np.ndarray      # [T] int64 class indices
    skill: int                # 0=N, 1=I, 2=E
    grs: Optional[float] = None


@dataclass
class JigsawsData:
    trials: List[Trial]
    gesture_names: List[str]                       # index -> "G1" ...
    feature_names: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return self.trials[0].kinematics.shape[1]

    @property
    def n_gestures(self) -> int:
        return len(self.gesture_names)

    @property
    def users(self) -> List[str]:
        return sorted({t.user for t in self.trials})


# --------------------------------------------------------------------------------------
# Feature selection
# --------------------------------------------------------------------------------------
def _block_names(prefix: str) -> List[str]:
    return ([f"{prefix}_pos_{a}" for a in "xyz"]
            + [f"{prefix}_rot_{i}{j}" for i in range(3) for j in range(3)]
            + [f"{prefix}_vel_{a}" for a in "xyz"]
            + [f"{prefix}_angvel_{a}" for a in "xyz"]
            + [f"{prefix}_gripper"])


ALL_FEATURE_NAMES = [n for m in MANIPULATORS for n in _block_names(m)]


def feature_columns(manipulators: Sequence[str] = ("PSM_L", "PSM_R")) -> List[int]:
    """Column indices for the selected manipulators (PSM-only is the common choice)."""
    cols: List[int] = []
    for m in manipulators:
        start = MANIPULATORS[m] * BLOCK
        cols.extend(range(start, start + BLOCK))
    return cols


# --------------------------------------------------------------------------------------
# Real JIGSAWS loader
# --------------------------------------------------------------------------------------
def _parse_meta(meta_path: Path) -> Dict[str, Tuple[int, Optional[float]]]:
    out: Dict[str, Tuple[int, Optional[float]]] = {}
    for line in meta_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        name, skill = parts[0], parts[1]
        if skill not in SKILL_TO_IDX:
            continue
        grs = float(parts[2]) if len(parts) > 2 else None
        out[name] = (SKILL_TO_IDX[skill], grs)
    return out


def _parse_transcription(path: Path) -> List[Tuple[int, int, str]]:
    segs = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3:
            segs.append((int(parts[0]), int(parts[1]), parts[2]))
    return segs


def load_jigsaws_task(task_dir: str | os.PathLike,
                      manipulators: Sequence[str] = ("PSM_L", "PSM_R")) -> JigsawsData:
    """Load one JIGSAWS task folder. Unlabelled frames (outside transcriptions) are dropped,
    which is the standard protocol for gesture recognition on JIGSAWS."""
    task_dir = Path(task_dir)
    task = task_dir.name
    meta_files = list(task_dir.glob("meta_file_*.txt"))
    if not meta_files:
        raise FileNotFoundError(f"No meta_file_*.txt in {task_dir}")
    meta = _parse_meta(meta_files[0])
    kin_dir = task_dir / "kinematics" / "AllGestures"
    tr_dir = task_dir / "transcriptions"
    cols = feature_columns(manipulators)

    raw: List[Tuple[str, str, np.ndarray, List[Tuple[int, int, str]]]] = []
    gesture_set = set()
    for tr_path in sorted(tr_dir.glob("*.txt")):
        name = tr_path.stem
        m = TRIAL_RE.match(name)
        kin_path = kin_dir / f"{name}.txt"
        if m is None or name not in meta or not kin_path.exists():
            continue
        kin = np.loadtxt(kin_path, dtype=np.float32, ndmin=2)
        if kin.shape[1] != N_KIN_COLS:
            raise ValueError(f"{kin_path}: expected {N_KIN_COLS} columns, got {kin.shape[1]}")
        segs = _parse_transcription(tr_path)
        gesture_set.update(g for _, _, g in segs)
        raw.append((name, m.group("user"), kin[:, cols], segs))

    if not raw:
        raise RuntimeError(f"No usable trials found in {task_dir}")

    gesture_names = sorted(gesture_set, key=lambda g: int(g[1:]))
    g2i = {g: i for i, g in enumerate(gesture_names)}
    trials: List[Trial] = []
    for name, user, kin, segs in raw:
        idx, labels = [], []
        for start, end, g in segs:
            s, e = max(start - 1, 0), min(end, len(kin))   # 1-indexed inclusive -> python slice
            idx.extend(range(s, e))
            labels.extend([g2i[g]] * (e - s))
        idx_arr = np.asarray(idx, dtype=np.int64)
        skill, grs = meta[name]
        trials.append(Trial(name=name, task=task, user=user,
                            kinematics=kin[idx_arr].astype(np.float32),
                            gestures=np.asarray(labels, dtype=np.int64),
                            skill=skill, grs=grs))
    feat_names = [ALL_FEATURE_NAMES[c] for c in cols]
    return JigsawsData(trials=trials, gesture_names=gesture_names, feature_names=feat_names)


# --------------------------------------------------------------------------------------
# Synthetic JIGSAWS-format generator (for testing the pipeline without the real data)
# --------------------------------------------------------------------------------------
SYNTH_USERS = {"B": "N", "C": "I", "D": "E", "E": "E", "F": "I", "G": "N", "H": "N", "I": "N"}
SYNTH_GESTURES = ["G1", "G2", "G3", "G4", "G5", "G6", "G8", "G9", "G10", "G11"]  # Suturing set


def generate_synthetic_jigsaws(out_dir: str | os.PathLike, task: str = "Suturing",
                               trials_per_user: int = 5, seed: int = 0) -> Path:
    """Write a mock task folder in the real JIGSAWS file layout.

    Each gesture has its own smooth velocity profile for the PSMs; skill level controls
    tremor noise, speed and how often gestures are repeated, so both tasks are learnable
    but not trivial. This is ONLY for pipeline testing; numbers from it mean nothing.
    """
    rng = np.random.default_rng(seed)
    task_dir = Path(out_dir) / task
    (task_dir / "kinematics" / "AllGestures").mkdir(parents=True, exist_ok=True)
    (task_dir / "transcriptions").mkdir(parents=True, exist_ok=True)

    n_g = len(SYNTH_GESTURES)
    protos = rng.normal(0, 1, size=(n_g, 2, 6))          # per gesture, per PSM: vel(3)+angvel(3)
    freqs = rng.uniform(0.5, 2.0, size=(n_g,))
    grippers = rng.uniform(-0.5, 1.0, size=(n_g, 2))
    skill_cfg = {"N": dict(noise=0.55, speed=0.7, extra=4, grs=(8, 15)),
                 "I": dict(noise=0.35, speed=0.85, extra=2, grs=(14, 22)),
                 "E": dict(noise=0.18, speed=1.0, extra=0, grs=(22, 30))}
    base_order = list(range(n_g))
    meta_lines = []

    for user, skill in SYNTH_USERS.items():
        cfg = skill_cfg[skill]
        user_bias = rng.normal(0, 0.3, size=(N_KIN_COLS,))
        for rep in range(1, trials_per_user + 1):
            name = f"{task}_{user}{rep:03d}"
            order = list(base_order)
            for _ in range(cfg["extra"] + rng.integers(0, 2)):
                order.insert(rng.integers(1, len(order)), rng.integers(0, n_g))
            frames, segs, t0 = [], [], 1
            pos = np.zeros((2, 3))
            for g in order:
                dur = int(rng.integers(25, 70) / cfg["speed"])
                t = np.arange(dur) / FRAME_RATE_HZ
                block = np.zeros((dur, N_KIN_COLS), dtype=np.float32)
                for k, m in enumerate(("PSM_L", "PSM_R")):
                    base = MANIPULATORS[m] * BLOCK
                    env = np.sin(2 * np.pi * freqs[g] * t)[:, None]
                    vel = env * protos[g, k, :3] * cfg["speed"]
                    ang = env * protos[g, k, 3:] * cfg["speed"]
                    vel = vel + rng.normal(0, cfg["noise"], vel.shape)
                    ang = ang + rng.normal(0, cfg["noise"], ang.shape)
                    p = pos[k] + np.cumsum(vel, axis=0) / FRAME_RATE_HZ
                    pos[k] = p[-1]
                    block[:, base:base + 3] = p
                    block[:, base + 3:base + 12] = np.eye(3).ravel() + rng.normal(0, 0.05, (dur, 9))
                    block[:, base + 12:base + 15] = vel
                    block[:, base + 15:base + 18] = ang
                    block[:, base + 18] = grippers[g, k] + rng.normal(0, cfg["noise"] * 0.3, dur)
                # master manipulators: noisy scaled copy of the PSMs
                block[:, 0:2 * BLOCK] = 3.0 * block[:, 2 * BLOCK:] + rng.normal(0, 0.1, (dur, 2 * BLOCK))
                block += user_bias
                frames.append(block)
                segs.append((t0, t0 + dur - 1, SYNTH_GESTURES[g]))
                t0 += dur
            kin = np.concatenate(frames)
            # a few unlabelled frames at the start/end, as in the real data
            pad = rng.normal(0, 0.1, (10, N_KIN_COLS)).astype(np.float32)
            kin = np.concatenate([pad, kin, pad])
            segs = [(s + 10, e + 10, g) for s, e, g in segs]
            np.savetxt(task_dir / "kinematics" / "AllGestures" / f"{name}.txt", kin, fmt="%.6f")
            (task_dir / "transcriptions" / f"{name}.txt").write_text(
                "\n".join(f"{s} {e} {g} " for s, e, g in segs) + "\n")
            grs = int(rng.integers(*cfg["grs"]))
            subs = " ".join(str(int(rng.integers(1, 6))) for _ in range(6))
            meta_lines.append(f"{name}\t\t{skill}\t\t{grs}\t\t{subs}")
    (task_dir / f"meta_file_{task}.txt").write_text("\n".join(meta_lines) + "\n")
    return task_dir


# --------------------------------------------------------------------------------------
# LOUO splitting and normalisation
# --------------------------------------------------------------------------------------
def louo_splits(data: JigsawsData) -> Iterator[Tuple[str, List[Trial], List[Trial]]]:
    """Leave-One-User-Out: every fold holds out ALL trials of one surgeon.
    No surgeon ever appears in both train and test, which prevents identity leakage."""
    for user in data.users:
        train = [t for t in data.trials if t.user != user]
        test = [t for t in data.trials if t.user == user]
        assert not ({t.user for t in train} & {t.user for t in test})
        yield user, train, test


def supertrial(t: Trial) -> int:
    """Repetition index of a JIGSAWS trial (``Suturing_B003`` -> 3)."""
    m = TRIAL_RE.match(t.name)
    if m is None:
        raise ValueError(f"{t.name}: not a JIGSAWS trial name")
    return int(m.group("rep"))


def loso_supertrial_splits(data: JigsawsData) -> Iterator[Tuple[str, List[Trial], List[Trial]]]:
    """Leave-One-Supertrial-Out (Gao et al., 2014; Ahmidi et al., 2017): fold k holds out the k-th
    repetition of EVERY surgeon. This is the protocol most JIGSAWS gesture-recognition papers
    report; surgeons appear in both training and test, so it measures generalisation to new
    repetitions, not to new surgeons (that is LOUO)."""
    reps = sorted({supertrial(t) for t in data.trials})
    for r in reps:
        train = [t for t in data.trials if supertrial(t) != r]
        test = [t for t in data.trials if supertrial(t) == r]
        assert not ({t.name for t in train} & {t.name for t in test})
        yield f"ST{r}", train, test


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, trials: Sequence[Trial]) -> "Normalizer":
        x = np.concatenate([t.kinematics for t in trials])
        return cls(mean=x.mean(0), std=x.std(0) + 1e-6)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


class TrialDataset(Dataset):
    """Full-length trials (one sample = one trial); statistics fitted on the TRAIN fold only."""

    def __init__(self, trials: Sequence[Trial], norm: Normalizer, downsample: int = 1):
        self.items = []
        for t in trials:
            x = norm(t.kinematics)[::downsample]
            y = t.gestures[::downsample]
            self.items.append((torch.from_numpy(x), torch.from_numpy(y), t.skill, t.name))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_padded(batch):
    """Pad variable-length trials to [B, T_max, F]; mask marks real frames; padded labels = -100."""
    xs, ys, skills, names = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    T, F = int(lengths.max()), xs[0].shape[1]
    X = torch.zeros(len(xs), T, F)
    Y = torch.full((len(xs), T), -100, dtype=torch.long)
    M = torch.zeros(len(xs), T, dtype=torch.bool)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :len(x)], Y[i, :len(y)], M[i, :len(x)] = x, y, True
    return X, Y, torch.tensor(skills), M, list(names)
