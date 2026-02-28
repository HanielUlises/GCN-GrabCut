"""
Visualisation utilities for GCN-GrabCut research.

Functions
---------
plot_training_curves   — loss/acc/IoU over epochs
plot_trimap_comparison — side-by-side input / trimap / result
plot_superpixel_graph  — draw graph on image using networkx + matplotlib
plot_attention_map     — visualise per-superpixel class probabilities
plot_confusion_matrix  — per-class confusion matrix for trimap labels
save_research_report   — save a grid image summarising an experiment
"""

from __future__ import annotations

import numpy as np
import cv2
from pathlib import Path
from typing import Optional

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import ListedColormap
    _MPL = True
except ImportError:
    _MPL = False


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
    show: bool = False,
) -> Optional["plt.Figure"]:
    """
    Plot train/val loss, val accuracy, and per-class IoU over epochs.

    Parameters
    
    history  : dict returned by Trainer.fit()
    save_path: if provided, save the figure to this path
    """
    if not _MPL:
        print("[visualise] matplotlib not installed — skipping plots.")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("GCN-GrabCut Training Curves", fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train", color="royalblue", linewidth=2)
    if history.get("val_loss"):
        ax.plot(range(1, len(history["val_loss"]) + 1),
                history["val_loss"], label="Val", color="tomato", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[1]
    if history.get("val_acc"):
        ax.plot(range(1, len(history["val_acc"]) + 1),
                history["val_acc"], color="seagreen", linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Validation Accuracy"); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Per-class IoU
    ax = axes[2]
    for key, label, color in [
        ("val_iou_bg",  "IoU BG",  "steelblue"),
        ("val_iou_unk", "IoU UNK", "goldenrod"),
        ("val_iou_fg",  "IoU FG",  "tomato"),
    ]:
        if history.get(key):
            ax.plot(range(1, len(history[key]) + 1),
                    history[key], label=label, color=color, linewidth=2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("IoU")
    ax.set_title("Per-Class IoU"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # LR
    if history.get("lr"):
        ax2 = axes[0].twinx()
        ax2.plot(epochs, history["lr"], color="grey", linestyle="--",
                 linewidth=1, alpha=0.5, label="LR")
        ax2.set_ylabel("LR", color="grey")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[visualise] Training curves saved → {save_path}")
    if show:
        plt.show()
    return fig

def plot_trimap_comparison(
    image:       np.ndarray,
    trimap:      np.ndarray,
    binary_mask: np.ndarray,
    gt_mask:     Optional[np.ndarray] = None,
    probs:       Optional[np.ndarray] = None,
    title:       str = "",
    save_path:   Optional[str] = None,
    show:        bool = False,
) -> Optional["plt.Figure"]:
    """
    Side-by-side panel: Input | Trimap | Result | [GT] | [FG probability heatmap]

    Parameters
    ----------
    image       : BGR original
    trimap      : (H, W) uint8 {0,1,2,3}
    binary_mask : (H, W) uint8 {0, 1}
    gt_mask     : optional GT binary mask
    probs       : optional (H, W) float array of per-pixel FG probability
    """
    if not _MPL:
        return None

    from .grabcut import Label

    n_panels = 3
    if gt_mask is not None:  n_panels += 1
    if probs   is not None:  n_panels += 1

    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    fig.suptitle(title or "GCN-GrabCut Result", fontsize=12)

    ax_idx = 0

    # Input
    axes[ax_idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[ax_idx].set_title("Input Image"); axes[ax_idx].axis("off")
    ax_idx += 1

    # Trimap
    trimap_rgb = _colour_trimap_rgb(trimap)
    axes[ax_idx].imshow(trimap_rgb)
    axes[ax_idx].set_title("Predicted Trimap")
    axes[ax_idx].axis("off")

    legend_patches = [
        mpatches.Patch(color=[0,0,0],     label="BG (definite)"),
        mpatches.Patch(color=[1,1,1],     label="FG (definite)"),
        mpatches.Patch(color=[0.2,0,0],   label="PROB BG"),
        mpatches.Patch(color=[0,0.8,0.8], label="PROB FG"),
    ]
    axes[ax_idx].legend(handles=legend_patches, loc="lower right", fontsize=6)
    ax_idx += 1

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    fgmask = binary_mask.astype(np.float32)[:, :, np.newaxis]
    green  = np.zeros_like(rgb); green[:, :, 1] = 200
    overlay = np.clip(rgb * (1 - 0.45 * fgmask) + green * 0.45 * fgmask, 0, 255).astype(np.uint8)
    axes[ax_idx].imshow(overlay); axes[ax_idx].set_title("Segmentation Result"); axes[ax_idx].axis("off")
    ax_idx += 1

    if gt_mask is not None:
        axes[ax_idx].imshow(gt_mask, cmap="gray"); axes[ax_idx].set_title("GT Mask"); axes[ax_idx].axis("off")
        ax_idx += 1

    if probs is not None:
        im = axes[ax_idx].imshow(probs, cmap="RdYlGn", vmin=0, vmax=1)
        axes[ax_idx].set_title("FG Probability"); axes[ax_idx].axis("off")
        plt.colorbar(im, ax=axes[ax_idx], fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def _colour_trimap_rgb(trimap: np.ndarray) -> np.ndarray:
    from .grabcut import Label
    vis = np.zeros((*trimap.shape, 3), dtype=np.float32)
    vis[trimap == Label.BG_DEFINITE] = [0.0, 0.0, 0.0]
    vis[trimap == Label.FG_DEFINITE] = [1.0, 1.0, 1.0]
    vis[trimap == Label.BG_PROBABLE] = [0.2, 0.0, 0.0]
    vis[trimap == Label.FG_PROBABLE] = [0.0, 0.8, 0.8]
    return vis

def plot_superpixel_graph(
    image:    np.ndarray,
    graph,
    probs:    Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    show:     bool = False,
    max_nodes: int = 500,
) -> Optional["plt.Figure"]:
    """
    Draw the superpixel graph over the image.
    Nodes are coloured by FG probability if provided.

    Parameters
    ----------
    image    : BGR image
    graph    : SuperpixelGraph
    probs    : (N, 3) softmax probabilities; nodes coloured by FG prob if given
    max_nodes: skip edge drawing for very large graphs (performance)
    """
    if not _MPL:
        return None

    import networkx as nx

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Superpixel Graph", fontsize=12)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from skimage.segmentation import mark_boundaries
    img_b = mark_boundaries(rgb, graph.segments, color=(1, 0.4, 0))
    axes[0].imshow(img_b); axes[0].set_title(f"Superpixels (N={graph.n_nodes})"); axes[0].axis("off")

    axes[1].imshow(rgb, alpha=0.4)
    H, W = image.shape[:2]

    if probs is not None:
        node_colors = probs[:, 2]
        cmap = plt.cm.RdYlGn
    else:
        node_colors = np.ones(graph.n_nodes) * 0.5
        cmap = plt.cm.Blues

    centroids = graph.node_centroids   # (N, 2) — [y_norm, x_norm]
    pos       = {i: (centroids[i, 1] * W, centroids[i, 0] * H) for i in range(graph.n_nodes)}

    n_edges = graph.edge_index.shape[1] // 2
    draw_edges = min(n_edges, max_nodes * 3)
    for i in range(0, n_edges * 2, 2)[:draw_edges]:
        s, d = graph.edge_index[0, i], graph.edge_index[1, i]
        if s < graph.n_nodes and d < graph.n_nodes:
            xs = [pos[s][0], pos[d][0]]
            ys = [pos[s][1], pos[d][1]]
            axes[1].plot(xs, ys, color="white", alpha=0.15, linewidth=0.5, zorder=1)

    xs = [pos[i][0] for i in range(graph.n_nodes)]
    ys = [pos[i][1] for i in range(graph.n_nodes)]
    sc = axes[1].scatter(xs, ys, c=node_colors, cmap=cmap,
                         vmin=0, vmax=1, s=20, zorder=2, edgecolors="none")
    plt.colorbar(sc, ax=axes[1], fraction=0.03, label="FG prob" if probs is not None else "")
    axes[1].set_title(f"Graph (N={graph.n_nodes}, E={graph.n_edges//2})"); axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig

def plot_confusion_matrix(
    preds:     np.ndarray,
    labels:    np.ndarray,
    class_names: list[str] = ["BG", "UNK", "FG"],
    save_path:  Optional[str] = None,
    show:       bool = False,
) -> Optional["plt.Figure"]:
    """Per-class confusion matrix for trimap label predictions."""
    if not _MPL:
        return None

    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for p, g in zip(preds.ravel(), labels.ravel()):
        if 0 <= p < n and 0 <= g < n:
            cm[g, p] += 1

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (trimap labels)")

    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def save_research_report(
    samples_results: list[dict],
    save_path:       str,
    title:           str = "GCN-GrabCut Experiment Results",
    max_rows:        int = 8,
) -> None:
    """
    Save a grid image showing input / trimap / result for multiple samples.

    Parameters
    ----------
    samples_results : list of dicts with keys:
        "image", "trimap", "binary_mask", optional "gt_mask", optional "name"
    save_path : output image path
    """
    if not _MPL:
        _save_cv_grid(samples_results, save_path, max_rows)
        return

    n = min(len(samples_results), max_rows)
    cols = 3 + (1 if "gt_mask" in samples_results[0] else 0)

    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 3.5 * n))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Input", "Predicted Trimap", "Segmentation Result"]
    if "gt_mask" in samples_results[0]:
        col_titles.append("Ground Truth")

    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold")

    for i in range(n):
        r = samples_results[i]
        img  = cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB)
        tri  = _colour_trimap_rgb(r["trimap"])
        fgm  = r["binary_mask"].astype(np.float32)[:, :, np.newaxis]
        ovl  = np.clip(img.astype(np.float32) * (1 - 0.4 * fgm) +
                       np.array([0, 200, 0]) * 0.4 * fgm, 0, 255).astype(np.uint8)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(tri)
        axes[i, 2].imshow(ovl)
        if "gt_mask" in r:
            axes[i, 3].imshow(r["gt_mask"], cmap="gray")

        name = r.get("name", f"sample_{i}")
        axes[i, 0].set_ylabel(name, fontsize=8)
        for j in range(cols):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"[visualise] Research report saved → {save_path}")
    plt.close(fig)


def _save_cv_grid(results: list[dict], path: str, max_rows: int) -> None:
    """Fallback grid saver using OpenCV."""
    rows = []
    for r in results[:max_rows]:
        img = cv2.resize(r["image"], (200, 150))
        msk = cv2.cvtColor(r["binary_mask"] * 255, cv2.COLOR_GRAY2BGR)
        msk = cv2.resize(msk, (200, 150))
        rows.append(np.concatenate([img, msk], axis=1))
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(path, grid)
