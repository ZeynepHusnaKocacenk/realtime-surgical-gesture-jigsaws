"""Daphnet Freezing of Gait (Bachlin et al., 2010; UCI Machine Learning Repository).

Ten Parkinson's patients wore three 3-axis accelerometers (ankle, upper leg, trunk) sampled at
64 Hz. Raw file columns: time_ms, ankle xyz, thigh xyz, trunk xyz, annotation where
0 = not part of the experiment, 1 = walking without freeze, 2 = freeze.

The loader returns the same ``JigsawsData`` container as the JIGSAWS loader:
  * a "trial" is one contiguous, labelled stretch of a recording (frames with annotation 0 are
    dropped, and a recording is split wherever such frames interrupt it, so no sequence ever
    spans a gap);
  * ``Trial.user`` is the subject id, ``Trial.gestures`` holds 0 = no freeze, 1 = freeze;
  * ``Trial.skill`` is unused (0).
Data is downsampled by averaging ``downsample`` consecutive samples, which is causal.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np

from .data import JigsawsData, Trial

SAMPLING_HZ = 64.0
GESTURE_NAMES = ["no_freeze", "freeze"]
FEATURE_NAMES = [f"{s}_{a}" for s in ("ankle", "thigh", "trunk") for a in ("x", "y", "z")]
ANKLE_VERTICAL = 1          # feature index of the ankle accelerometer's vertical axis (y)
_FILE_RE = re.compile(r"^(S\d+)R(\d+)\.txt$")


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """[start, end) of every run of True in a boolean vector."""
    if not len(mask):
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def load_daphnet(data_dir: str | Path, downsample: int = 2, min_frames: int = 16) -> JigsawsData:
    """Load every ``SxxRyy.txt`` below ``data_dir``. Effective rate = 64 / downsample Hz."""
    files = sorted(p for p in Path(data_dir).rglob("S*R*.txt") if _FILE_RE.match(p.name))
    if not files:
        raise FileNotFoundError(f"no Daphnet files (SxxRyy.txt) found under {data_dir}")
    trials: List[Trial] = []
    for path in files:
        subject = _FILE_RE.match(path.name).group(1)
        raw = np.loadtxt(path, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != 11:
            raise ValueError(f"{path.name}: expected 11 columns, got shape {raw.shape}")
        x, ann = raw[:, 1:10], raw[:, 10].astype(int)
        if not set(np.unique(ann)) <= {0, 1, 2}:
            raise ValueError(f"{path.name}: unexpected annotation values {np.unique(ann)}")
        for k, (s, e) in enumerate(_runs(ann != 0)):          # drop "not part of experiment"
            seg_x, seg_y = x[s:e], (ann[s:e] == 2).astype(np.int64)
            n = (len(seg_x) // downsample) * downsample
            if n // downsample < min_frames:
                continue
            xs = seg_x[:n].reshape(-1, downsample, 9).mean(1)   # causal averaging
            ys = seg_y[:n].reshape(-1, downsample)[:, -1]       # label of the newest sample
            trials.append(Trial(name=f"{path.stem}_seg{k}", task="Daphnet", user=subject,
                                kinematics=xs.astype(np.float32), gestures=ys, skill=0))
    return JigsawsData(trials=trials, gesture_names=list(GESTURE_NAMES),
                       feature_names=list(FEATURE_NAMES))


def has_freeze(trials) -> bool:
    return any(int(t.gestures.max()) == 1 for t in trials)


def loso_splits(data: JigsawsData, n_val: int = 2, seed: int = 0
                ) -> Iterator[Tuple[str, List[Trial], List[Trial], List[Trial]]]:
    """Leave-one-subject-out with a validation set carved out of the training subjects.

    Yields (test_subject, train_trials, val_trials, test_trials). Test subjects without any freeze
    episode are skipped (no event-level metric exists for them). ``n_val`` validation subjects are
    drawn (by ``seed``) from the remaining subjects that DO have freezes; thresholds and epoch
    selection use them, never the test subject. Subjects never overlap across the three sets.
    """
    by_user = {u: [t for t in data.trials if t.user == u] for u in data.users}
    rng = np.random.default_rng(seed)
    for user in data.users:
        if not has_freeze(by_user[user]):
            continue
        others = [u for u in data.users if u != user]
        cands = [u for u in others if has_freeze(by_user[u])]
        val_users = sorted(rng.choice(cands, size=min(n_val, len(cands) - 1), replace=False).tolist())
        train = [t for u in others if u not in val_users for t in by_user[u]]
        val = [t for u in val_users for t in by_user[u]]
        test = by_user[user]
        assert not ({t.user for t in train} & {t.user for t in test})
        assert not ({t.user for t in val} & ({t.user for t in test} | {t.user for t in train}))
        yield user, train, val, test


# ---------------------------------------------------------------------------------------
# Freeze Index baseline (Moore et al., 2008)
# ---------------------------------------------------------------------------------------
def freeze_index(x: np.ndarray, rate_hz: float, window_s: float = 4.0,
                 loco_band=(0.5, 3.0), freeze_band=(3.0, 8.0)) -> Tuple[np.ndarray, np.ndarray]:
    """Causal Freeze Index of a 1-D leg accelerometer signal.

    For every frame t, take the last ``window_s`` seconds, remove the mean, and compute
    FI = power in the freeze band (3-8 Hz) / power in the locomotor band (0.5-3 Hz).
    Returns (FI, total power in 0.5-8 Hz); both are 0 until a full window is available.
    """
    n = int(round(window_s * rate_hz))
    fi, power = np.zeros(len(x)), np.zeros(len(x))
    if len(x) < n:
        return fi, power
    freqs = np.fft.rfftfreq(n, 1.0 / rate_hz)
    loco = (freqs > loco_band[0]) & (freqs <= loco_band[1])
    frz = (freqs > freeze_band[0]) & (freqs <= freeze_band[1])
    win = np.lib.stride_tricks.sliding_window_view(np.asarray(x, dtype=np.float64), n)
    for s in range(0, len(win), 8192):                       # chunked to bound memory
        w = win[s:s + 8192]
        p = np.abs(np.fft.rfft(w - w.mean(1, keepdims=True), axis=1)) ** 2 / n
        pl, pf = p[:, loco].sum(1), p[:, frz].sum(1)
        fi[n - 1 + s: n - 1 + s + len(w)] = pf / (pl + 1e-12)
        power[n - 1 + s: n - 1 + s + len(w)] = pl + pf
    return fi, power
