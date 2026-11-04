"""Training loop for joint gesture + skill models (one LOUO fold at a time)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import SeqModel


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    skill_weight: float = 0.5      # weight of the skill loss relative to the gesture loss
    smooth_weight: float = 0.15    # MS-TCN temporal smoothing loss, reduces over-segmentation
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    gesture_loss: List[float] = field(default_factory=list)
    skill_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


def _final_skill_logits(skill_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Skill logits at each trial's last real frame (= prediction from the whole trial)."""
    last = mask.sum(1) - 1
    return skill_logits[torch.arange(len(last)), last]


def smoothing_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Truncated MSE between consecutive frame log-probabilities (Farha & Gall, 2019)."""
    lp = F.log_softmax(logits, dim=-1)
    diff = (lp[:, 1:] - lp[:, :-1].detach()) ** 2
    diff = torch.clamp(diff, max=16.0).mean(-1)
    pair_mask = (mask[:, 1:] & mask[:, :-1]).float()
    return (diff * pair_mask).sum() / pair_mask.sum().clamp(min=1.0)


def compute_loss(model: SeqModel, batch, cfg: TrainConfig) -> Dict[str, torch.Tensor]:
    X, Y, S, M, _ = batch
    X, Y, S, M = X.to(cfg.device), Y.to(cfg.device), S.to(cfg.device), M.to(cfg.device)
    g_logits, s_logits = model(X, M)
    g_loss = F.cross_entropy(g_logits.reshape(-1, g_logits.shape[-1]), Y.reshape(-1), ignore_index=-100)
    s_loss = F.cross_entropy(_final_skill_logits(s_logits, M), S)
    sm_loss = smoothing_loss(g_logits, M)
    total = g_loss + cfg.skill_weight * s_loss + cfg.smooth_weight * sm_loss
    return {"total": total, "gesture": g_loss, "skill": s_loss}


def train_model(model: SeqModel, train_loader: DataLoader, val_loader: DataLoader,
                cfg: TrainConfig, verbose: bool = False) -> History:
    """Trains for a FIXED number of epochs. The held-out surgeon's loss is logged for
    monitoring only; it is never used for early stopping or model selection, so the
    LOUO test fold stays unseen."""
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    hist = History()
    for epoch in range(cfg.epochs):
        model.train()
        tot = g = s = 0.0
        for batch in train_loader:
            losses = compute_loss(model, batch, cfg)
            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tot += losses["total"].item(); g += losses["gesture"].item(); s += losses["skill"].item()
        sched.step()
        n = len(train_loader)
        hist.train_loss.append(tot / n); hist.gesture_loss.append(g / n); hist.skill_loss.append(s / n)
        model.eval()
        with torch.no_grad():
            v = [compute_loss(model, b, cfg)["total"].item() for b in val_loader]
        hist.val_loss.append(sum(v) / max(len(v), 1))
        if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
            print(f"    epoch {epoch:3d}  train {hist.train_loss[-1]:.3f}  "
                  f"(gesture {hist.gesture_loss[-1]:.3f}, skill {hist.skill_loss[-1]:.3f})  "
                  f"held-out {hist.val_loss[-1]:.3f}")
    model.eval()
    return hist
