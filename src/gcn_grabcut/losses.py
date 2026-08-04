"""Shared loss functions for GCN-GrabCut."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .model import CLASS_BG, CLASS_UNK, CLASS_FG


class FocalLoss(nn.Module):
    """
    Focal Loss — FL(p) = -α(1-p)^γ · log(p)

    Downweights easy examples so the model focuses on hard ones.
    gamma=2.0 is the original paper default (Lin et al. 2017).
    Raise to 2.5 only if UNKNOWN class dominates heavily.
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
    """
    Cross-entropy with label smoothing.
    Reduces overconfidence — useful when trimap labels near boundaries are noisy.
    """

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


class TrimapLoss(nn.Module):
    """
    Region-level objective aligned with the pixel-level segmentation metric.

    Training on superpixels introduces a mismatch: the loss counts regions,
    while the metric counts pixels, so a large region and a sliver of a region
    carry equal weight even though they cost very different amounts of IoU.
    Two terms remove that mismatch.

    Classification term
        Focal cross-entropy over {BG, UNKNOWN, FG}, weighted by region area,
        so a mistake is penalised in proportion to the image it covers.

    Overlap term
        A soft Dice score on the expected foreground coverage
        ``p_i = P(FG) + ½·P(UNKNOWN)``, accumulated with area weights. This is
        the region-space equivalent of pixel Dice, and unlike cross-entropy it
        is driven by the shape of the whole mask rather than by independent
        per-region decisions. When ground-truth coverage ratios are supplied
        the target is the true fraction, so regions straddling the object
        boundary contribute a graded rather than a thresholded signal.

    Parameters
    ----------
    gamma        : focal exponent; 0 recovers plain cross-entropy
    weight       : per-class weights (length 3)
    dice_weight  : relative weight of the overlap term
    area_weighted: weight the classification term by region area
    """

    def __init__(
        self,
        gamma:         float = 2.0,
        weight:        Optional[torch.Tensor] = None,
        dice_weight:   float = 0.5,
        area_weighted: bool  = True,
        eps:           float = 1e-6,
    ):
        super().__init__()
        self.gamma         = gamma
        self.weight        = weight
        self.dice_weight   = dice_weight
        self.area_weighted = area_weighted
        self.eps           = eps

    def forward(
        self,
        logits:   torch.Tensor,
        labels:   torch.Tensor,
        area:     Optional[torch.Tensor] = None,
        fg_ratio: Optional[torch.Tensor] = None,
        batch:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ce  = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
        p_t = torch.exp(-ce.detach().clamp(max=30.0)) if self.gamma > 0 else None
        per_node = ce if p_t is None else ((1 - p_t) ** self.gamma) * ce

        if area is not None and self.area_weighted:
            w = area.to(per_node.dtype)
            w = w * (w.numel() / w.sum().clamp(min=self.eps))
            cls_loss = (per_node * w).mean()
        else:
            cls_loss = per_node.mean()

        if self.dice_weight <= 0:
            return cls_loss

        probs = F.softmax(logits, dim=-1)
        pred  = probs[:, CLASS_FG] + 0.5 * probs[:, CLASS_UNK]

        if fg_ratio is not None:
            target = fg_ratio.to(pred.dtype)
        else:
            target = ((labels == CLASS_FG).to(pred.dtype)
                      + 0.5 * (labels == CLASS_UNK).to(pred.dtype))

        a = torch.ones_like(pred) if area is None else area.to(pred.dtype)

        if batch is None:
            inter = (a * pred * target).sum()
            denom = (a * pred).sum() + (a * target).sum()
            dice  = 1.0 - (2.0 * inter + self.eps) / (denom + self.eps)
        else:
            # Overlap is a per-image ratio, so it is accumulated per graph and
            # averaged; pooling the whole batch would let a large image mask
            # the error made on a small one.
            n_graphs = int(batch.max().item()) + 1
            zeros = torch.zeros(n_graphs, device=pred.device, dtype=pred.dtype)
            inter = zeros.clone().index_add_(0, batch, a * pred * target)
            sum_p = zeros.clone().index_add_(0, batch, a * pred)
            sum_t = zeros.clone().index_add_(0, batch, a * target)
            dice  = (1.0 - (2.0 * inter + self.eps) / (sum_p + sum_t + self.eps)).mean()

        return cls_loss + self.dice_weight * dice