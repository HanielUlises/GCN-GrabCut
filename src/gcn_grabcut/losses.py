"""Shared loss functions for GCN-GrabCut."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss — FL(p) = -α(1-p)^γ · log(p)
    gamma=2.0 is the original paper default.
    Increase to 2.5 only if UNKNOWN class dominates heavily.
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        p_t   = torch.exp(-ce)
        return (((1 - p_t) ** self.gamma) * ce).mean()


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing — reduces overconfidence."""
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth * log_probs).sum(dim=-1)
        if self.weight is not None:
            loss = loss * self.weight[labels]
        return loss.mean()