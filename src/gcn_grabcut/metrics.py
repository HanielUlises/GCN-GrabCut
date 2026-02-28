"""
Evaluation metrics for GCN-GrabCut.

Segmentation metrics (binary mask evaluation)
- IoU (Jaccard Index)
- Dice coefficient
- Precision / Recall / F1
- Pixel accuracy
- Boundary F1 (BF score)

Trimap quality metrics
- FG recall / precision
- BG recall / precision
- Unknown fraction
- Trimap accuracy vs GT

"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass


# Binary segmentation metrics

@dataclass
class SegmentationMetrics:
    iou:            float
    dice:           float
    precision:      float
    recall:         float
    f1:             float
    pixel_accuracy: float
    boundary_f1:    float = 0.0

    def __str__(self) -> str:
        return (
            f"IoU={self.iou:.4f}  Dice={self.dice:.4f}  "
            f"Prec={self.precision:.4f}  Rec={self.recall:.4f}  "
            f"F1={self.f1:.4f}  PixAcc={self.pixel_accuracy:.4f}  "
            f"BF1={self.boundary_f1:.4f}"
        )

    def as_dict(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "dice": round(self.dice, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "pixel_accuracy": round(self.pixel_accuracy, 4),
            "boundary_f1": round(self.boundary_f1, 4),
        }


def evaluate(
    pred:           np.ndarray,
    gt:             np.ndarray,
    boundary_width: int = 3,
) -> SegmentationMetrics:
    """
    Compute full segmentation metrics.

    Parameters
    ----------
    pred, gt       : binary (H, W) arrays. values {0, 1}
    boundary_width : morphological kernel radius for boundary F1

    Returns
    -------
    SegmentationMetrics
    """
    orig_shape = pred.shape
    p = pred.astype(bool).ravel()
    g = gt.astype(bool).ravel()

    tp = ( p &  g).sum()
    fp = ( p & ~g).sum()
    fn = (~p &  g).sum()
    tn = (~p & ~g).sum()

    iou       = float(tp / (tp + fp + fn + 1e-8))
    dice      = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))
    f1        = float(2 * precision * recall / (precision + recall + 1e-8))
    pix_acc   = float((tp + tn) / (tp + tn + fp + fn + 1e-8))

    bf1 = boundary_f1(
        pred.reshape(orig_shape).astype(np.uint8),
        gt.reshape(orig_shape).astype(np.uint8),
        width=boundary_width,
    ) if boundary_width > 0 else 0.0

    return SegmentationMetrics(
        iou=iou, dice=dice,
        precision=precision, recall=recall,
        f1=f1, pixel_accuracy=pix_acc,
        boundary_f1=bf1,
    )


def boundary_f1(
    pred_2d: np.ndarray,
    gt_2d:   np.ndarray,
    width:   int = 3,
) -> float:
    """
    Boundary F1 score measures alignment of predicted and GT boundaries.

    Uses morphological dilation to tolerate small shifts.
    """
    kernel  = np.ones((width * 2 + 1,) * 2, np.uint8)

    def get_boundary(m: np.ndarray) -> np.ndarray:
        eroded = cv2.erode(m, kernel)
        return (m - eroded).astype(bool).ravel()

    pred_b = get_boundary(pred_2d.astype(np.uint8))
    gt_b   = get_boundary(gt_2d.astype(np.uint8))

    tp   = (pred_b &  gt_b).sum()
    prec = float(tp / (pred_b.sum() + 1e-8))
    rec  = float(tp / (gt_b.sum()  + 1e-8))
    return float(2 * prec * rec / (prec + rec + 1e-8))


# Trimap quality metrics
@dataclass
class TrimapMetrics:
    fg_recall:           float
    fg_precision:        float
    bg_recall:           float
    bg_precision:        float
    bg_contamination:    float   # FG-labelled pixels that are actually BG
    unknown_fraction:    float
    trimap_accuracy:     float   # how much of the trimap matches the GT

    def __str__(self) -> str:
        return (
            f"FG_rec={self.fg_recall:.3f}  FG_prec={self.fg_precision:.3f}  "
            f"BG_rec={self.bg_recall:.3f}  BG_cont={self.bg_contamination:.3f}  "
            f"Unk={self.unknown_fraction:.3f}  Acc={self.trimap_accuracy:.3f}"
        )

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.__dict__.items()}


def evaluate_trimap(
    trimap:  np.ndarray,
    gt_mask: np.ndarray,
) -> TrimapMetrics:
    """
    Evaluate a predicted trimap against a binary GT mask.

    Parameters
    ----------
    trimap  : (H, W) uint8, values in {0=BG, 1=FG, 2=PROB_BG, 3=PROB_FG}
    gt_mask : (H, W) uint8, binary {0=BG, 1=FG}
    """
    from .grabcut import Label

    gt  = gt_mask.astype(bool)
    n   = gt.size

    pred_fg  = (trimap == Label.FG_DEFINITE)
    pred_bg  = (trimap == Label.BG_DEFINITE)
    pred_pfg = (trimap == Label.FG_PROBABLE)
    pred_pbg = (trimap == Label.BG_PROBABLE)

    # FG metrics
    fg_tp  = (pred_fg &  gt).sum()
    fg_fp  = (pred_fg & ~gt).sum()
    fg_fn  = (~pred_fg &  gt).sum()
    fg_rec  = float(fg_tp / (fg_tp + fg_fn + 1e-8))
    fg_prec = float(fg_tp / (fg_tp + fg_fp + 1e-8))

    # BG metrics
    bg_tp  = (pred_bg & ~gt).sum()
    bg_fp  = (pred_bg &  gt).sum()
    bg_fn  = (~pred_bg & ~gt).sum()
    bg_rec  = float(bg_tp / (bg_tp + bg_fn + 1e-8))
    bg_prec = float(bg_tp / (bg_tp + bg_fp + 1e-8))

    # Contamination and unknown
    bg_cont   = float(fg_fp / n)
    unk_frac  = float((pred_pfg | pred_pbg).sum() / n)

    # Overall accuracy (treat PROB_FG as FG, PROB_BG as BG)
    pred_bin  = (pred_fg | pred_pfg).astype(np.uint8)
    tri_acc   = float((pred_bin.ravel() == gt_mask.ravel()).mean())

    return TrimapMetrics(
        fg_recall=fg_rec, fg_precision=fg_prec,
        bg_recall=bg_rec, bg_precision=bg_prec,
        bg_contamination=bg_cont,
        unknown_fraction=unk_frac,
        trimap_accuracy=tri_acc,
    )

def evaluate_batch(
    results: list[dict],
) -> dict:
    """
    Aggregate metrics over a list of result dicts.
    Each dict must have "binary_mask" and "gt_mask" keys.

    Returns
    -------
    dict of mean ± std for each metric
    """
    all_iou, all_dice, all_bf1 = [], [], []

    for r in results:
        m = evaluate(r["binary_mask"], r["gt_mask"])
        all_iou.append(m.iou)
        all_dice.append(m.dice)
        all_bf1.append(m.boundary_f1)

    return {
        "mean_iou":  float(np.mean(all_iou)),
        "std_iou":   float(np.std(all_iou)),
        "mean_dice": float(np.mean(all_dice)),
        "std_dice":  float(np.std(all_dice)),
        "mean_bf1":  float(np.mean(all_bf1)),
        "std_bf1":   float(np.std(all_bf1)),
        "n":         len(results),
    }
