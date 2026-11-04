"""Sequence models for joint frame-wise gesture recognition and trial-level skill assessment.

Every model maps kinematics [B, T, F] to
  * gesture logits  [B, T, n_gestures]   (one prediction per frame)
  * skill logits    [B, T, n_skills]     (prediction from the trial PREFIX up to frame t,
                                          via cumulative mean pooling of frame features)

The key design axis is CAUSALITY:
  * causal models (causal TCN, uni-directional LSTM) only use frames <= t, so the offline
    output at frame t is exactly what a live system would output at time t;
  * acausal models (centered TCN, BiLSTM) look at future frames. They are the usual
    offline baselines, but in the operating room they must be re-run on the history
    seen so far, which costs accuracy and latency. Measuring that gap is the point.

Causal models also expose a streaming API (``stream_reset`` / ``stream_step``) that
processes one frame at a time; tests check it reproduces the offline outputs exactly.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_SKILLS = 3


def cumulative_mean(h: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Running mean over time of h [B, T, H], ignoring padded frames."""
    if mask is None:
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
    m = mask.unsqueeze(-1).to(h.dtype)
    csum = torch.cumsum(h * m, dim=1)
    cnt = torch.cumsum(m, dim=1).clamp(min=1.0)
    return csum / cnt


class SeqModel(nn.Module):
    """Base class: subclasses implement ``features`` -> [B, T, H]."""

    causal: bool = False
    name: str = "base"

    def __init__(self, hidden: int, n_gestures: int, n_skills: int = N_SKILLS):
        super().__init__()
        self.gesture_head = nn.Linear(hidden, n_gestures)
        self.skill_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_skills))

    def features(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x, mask)
        return self.gesture_head(h), self.skill_head(cumulative_mean(h, mask))

    # ---- streaming (causal models only) -------------------------------------------------
    def stream_reset(self) -> None:
        raise NotImplementedError(f"{self.name} is acausal and cannot stream frame-by-frame")

    def stream_step(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(f"{self.name} is acausal and cannot stream frame-by-frame")

    def _stream_pool(self, h_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._sum = h_t if self._sum is None else self._sum + h_t
        self._n += 1
        return self.gesture_head(h_t), self.skill_head(self._sum / self._n)


# ---------------------------------------------------------------------------------------
# Temporal Convolutional Network (single-stage MS-TCN style, dilated residual layers)
# ---------------------------------------------------------------------------------------
class DilatedResidual(nn.Module):
    def __init__(self, ch: int, dilation: int, kernel: int, causal: bool, dropout: float):
        super().__init__()
        self.causal, self.dilation, self.kernel = causal, dilation, kernel
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.out = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, T]
        pad = (self.kernel - 1) * self.dilation
        xp = F.pad(x, (pad, 0)) if self.causal else F.pad(x, (pad // 2, pad - pad // 2))
        return x + self.drop(self.out(F.relu(self.conv(xp))))


class TCN(SeqModel):
    def __init__(self, n_features: int, n_gestures: int, hidden: int = 64, layers: int = 8,
                 kernel: int = 3, dropout: float = 0.3, causal: bool = True):
        super().__init__(hidden, n_gestures)
        self.causal = causal
        self.name = "TCN-causal" if causal else "TCN-acausal"
        self.inp = nn.Conv1d(n_features, hidden, 1)
        self.layers = nn.ModuleList(
            DilatedResidual(hidden, 2 ** i, kernel, causal, dropout) for i in range(layers))
        self.receptive_field = 1 + sum((kernel - 1) * 2 ** i for i in range(layers))

    def features(self, x, mask=None):
        h = self.inp(x.transpose(1, 2))
        for layer in self.layers:
            h = layer(h)
            if mask is not None:
                h = h * mask.unsqueeze(1)
        return h.transpose(1, 2)

    def stream_reset(self):
        if not self.causal:
            return super().stream_reset()
        self._buf: deque = deque(maxlen=self.receptive_field)
        self._sum, self._n = None, 0

    @torch.no_grad()
    def stream_step(self, x_t):
        """x_t: [F]. Re-runs the TCN on the last ``receptive_field`` frames, which is exactly
        the information the offline causal output at this frame depends on."""
        if not self.causal:
            return super().stream_step(x_t)
        self._buf.append(x_t)
        window = torch.stack(tuple(self._buf)).unsqueeze(0)       # [1, w, F]
        h_t = self.features(window)[0, -1]
        return self._stream_pool(h_t)


# ---------------------------------------------------------------------------------------
# LSTM / BiLSTM
# ---------------------------------------------------------------------------------------
class LSTMModel(SeqModel):
    def __init__(self, n_features: int, n_gestures: int, hidden: int = 64, layers: int = 1,
                 dropout: float = 0.3, bidirectional: bool = False):
        super().__init__(hidden, n_gestures)
        self.causal = not bidirectional
        self.name = "BiLSTM" if bidirectional else "LSTM"
        self.inp = nn.Linear(n_features, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(hidden * (2 if bidirectional else 1), hidden)
        self.drop = nn.Dropout(dropout)

    def features(self, x, mask=None):
        z = F.relu(self.inp(x))
        if mask is not None and not self.causal:
            # pack so the backward direction does not start inside padding
            lengths = mask.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(z, lengths, batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        else:
            out, _ = self.lstm(z)
        return self.proj(self.drop(out))

    def stream_reset(self):
        if not self.causal:
            return super().stream_reset()
        self._state = None
        self._sum, self._n = None, 0

    @torch.no_grad()
    def stream_step(self, x_t):
        """Constant-cost update: carries the LSTM hidden state across frames."""
        if not self.causal:
            return super().stream_step(x_t)
        z = F.relu(self.inp(x_t.view(1, 1, -1)))
        out, self._state = self.lstm(z, self._state)
        h_t = self.proj(out)[0, -1]
        return self._stream_pool(h_t)


MODEL_NAMES = ["TCN-causal", "TCN-acausal", "LSTM", "BiLSTM"]


def build_model(name: str, n_features: int, n_gestures: int, hidden: int = 64,
                tcn_layers: int = 8, dropout: float = 0.3) -> SeqModel:
    if name == "TCN-causal":
        return TCN(n_features, n_gestures, hidden, tcn_layers, dropout=dropout, causal=True)
    if name == "TCN-acausal":
        return TCN(n_features, n_gestures, hidden, tcn_layers, dropout=dropout, causal=False)
    if name == "LSTM":
        return LSTMModel(n_features, n_gestures, hidden, dropout=dropout, bidirectional=False)
    if name == "BiLSTM":
        return LSTMModel(n_features, n_gestures, hidden, dropout=dropout, bidirectional=True)
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
