"""
GCN-GrabCut End-to-End Pipeline — fully automatic, no user interaction.

Orchestrates:
  1. Superpixel graph construction
  2. Automatic FG/BG prior computation (saliency + boundary connectivity)
  3. GCN inference → trimap
  4. GrabCut refinement → binary mask
  5. Mask clean-up (small-component removal, optional largest-object keep)

Usage
-----
pipeline = GCNGrabCutPipeline(model)
result   = pipeline.segment(image)
result.save("output")
"""

from __future__ import annotations

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional

from .grabcut import GrabCut, GrabCutConfig, Label
from .graph_builder import GraphBuilder, SuperpixelGraphConfig
from .metrics import evaluate, evaluate_trimap, SegmentationMetrics, TrimapMetrics
from .model import CLASS_BG, CLASS_FG, project_to_pixels


@dataclass
class SegmentationResult:
    """All outputs from one pipeline run."""
    image:         np.ndarray              # Original BGR
    binary_mask:   np.ndarray              # (H, W) uint8 {0, 1}
    trimap:        np.ndarray              # (H, W) uint8 {0,1,2,3}
    segments:      np.ndarray             # (H, W) superpixel map
    overlay:       np.ndarray              # BGR with coloured overlay
    rgba:          np.ndarray              # BGRA transparent background
    timing:        dict = field(default_factory=dict)

    def show(self) -> None:
        """Display result panels in a window (blocks until key press)."""
        trimap_vis = _colour_trimap(self.trimap)
        panel = np.concatenate([
            cv2.resize(self.image,    (256, 256)),
            cv2.resize(trimap_vis,    (256, 256)),
            cv2.resize(self.overlay,  (256, 256)),
        ], axis=1)
        cv2.imshow("Input | Trimap | Result", panel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, prefix: str = "result") -> None:
        cv2.imwrite(f"{prefix}_overlay.png",      self.overlay)
        cv2.imwrite(f"{prefix}_rgba.png",          self.rgba)
        cv2.imwrite(f"{prefix}_trimap_colour.png", _colour_trimap(self.trimap))
        cv2.imwrite(f"{prefix}_mask.png",          self.binary_mask * 255)
        print(f"Saved outputs with prefix: {prefix}")

    def evaluate_against(
        self, gt_mask: np.ndarray
    ) -> tuple[SegmentationMetrics, TrimapMetrics]:
        """Compute segmentation and trimap metrics against a GT mask."""
        seg_m    = evaluate(self.binary_mask, gt_mask)
        trimap_m = evaluate_trimap(self.trimap, gt_mask)
        return seg_m, trimap_m


def guided_filter(
    guide:  np.ndarray,
    src:    np.ndarray,
    radius: int   = 8,
    eps:    float = 1e-3,
) -> np.ndarray:
    """
    Edge-preserving filter of `src` under the structure of `guide`.

    The O(1) box-filter formulation of He et al. (2010), "Guided Image
    Filtering". Applied to a region-level probability map it acts as an
    edge-aware upsampler: values stay flat inside homogeneous areas and break
    exactly where the guide image has an edge.

    Parameters
    ----------
    guide  : (H, W) float32 in [0, 1] — usually the grey-level image
    src    : (H, W) float32 — signal to filter
    radius : box radius in pixels
    eps    : regularisation; larger values smooth across weaker edges
    """
    k = (2 * radius + 1, 2 * radius + 1)
    mean_g  = cv2.blur(guide, k)
    mean_s  = cv2.blur(src, k)
    cov_gs  = cv2.blur(guide * src, k) - mean_g * mean_s
    var_g   = cv2.blur(guide * guide, k) - mean_g * mean_g

    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    return cv2.blur(a, k) * guide + cv2.blur(b, k)


def refine_trimap(
    probs:        np.ndarray,
    segments:     np.ndarray,
    image:        np.ndarray,
    threshold_fg: float = 0.55,
    threshold_bg: float = 0.55,
    radius:       int   = 8,
    eps:          float = 1e-3,
) -> np.ndarray:
    """
    Turn per-region class probabilities into a pixel-level trimap whose
    boundaries follow image edges rather than superpixel edges.

    Thresholding region probabilities directly makes every trimap boundary a
    superpixel boundary, and GrabCut then inherits that quantisation: the
    unknown band is as coarse as the tessellation. Projecting the
    probabilities to pixels and filtering them under the image as guide moves
    the definite/unknown transitions onto the nearest real intensity edge,
    which is where the true object boundary lies, before any label decision
    is taken.

    Parameters
    ----------
    probs    : (N, 3) per-region class probabilities [BG, UNK, FG]
    segments : (H, W) region label map
    image    : (H, W, 3) BGR source image
    radius, eps : guided-filter parameters

    Returns
    -------
    trimap : (H, W) uint8 in OpenCV GrabCut label space
    """
    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    p_bg = project_to_pixels(probs[:, CLASS_BG].astype(np.float32), segments)
    p_fg = project_to_pixels(probs[:, CLASS_FG].astype(np.float32), segments)

    p_bg = np.clip(guided_filter(guide, p_bg, radius, eps), 0.0, 1.0)
    p_fg = np.clip(guided_filter(guide, p_fg, radius, eps), 0.0, 1.0)

    trimap = np.where(p_fg > p_bg, Label.FG_PROBABLE, Label.BG_PROBABLE).astype(np.uint8)
    trimap[p_bg >= threshold_bg] = Label.BG_DEFINITE
    trimap[p_fg >= threshold_fg] = Label.FG_DEFINITE
    return trimap


def _seed_from_prior(
    trimap:   np.ndarray,
    graph,
    seed_frac: float = 0.1,
) -> np.ndarray:
    """
    Guarantee the trimap contains both a foreground and a background seed.

    If the GCN labels every superpixel the same way there is nobody to click a
    correction, so the most confident superpixels of the automatic prior are
    promoted to the missing side.

    Parameters
    ----------
    trimap    : (H, W) uint8 predicted trimap
    graph     : the SuperpixelGraph the trimap was predicted from
    seed_frac : fraction of superpixels to promote (at least one)
    """
    prior = graph.prior_features
    if prior is None or prior.size == 0:
        return trimap

    has_fg = np.isin(trimap, (Label.FG_DEFINITE, Label.FG_PROBABLE)).any()
    has_bg = np.isin(trimap, (Label.BG_DEFINITE, Label.BG_PROBABLE)).any()
    if has_fg and has_bg:
        return trimap

    n_seed = max(1, int(round(seed_frac * graph.n_nodes)))
    trimap = trimap.copy()

    if not has_fg:
        ids = np.argsort(prior[:, 0])[::-1][:n_seed]
        trimap[np.isin(graph.segments, ids)] = Label.FG_PROBABLE
    if not has_bg:
        ids = np.argsort(prior[:, 1])[::-1][:n_seed]
        trimap[np.isin(graph.segments, ids)] = Label.BG_PROBABLE

    return trimap


def clean_mask(
    mask:           np.ndarray,
    min_area_ratio: float = 0.002,
    keep_largest:   bool  = False,
) -> np.ndarray:
    """
    Remove spurious connected components from an automatic mask.

    Parameters
    ----------
    mask           : (H, W) uint8 {0, 1}
    min_area_ratio : components smaller than this fraction of the image are
                     dropped (0 disables the filter)
    keep_largest   : if True, keep only the single largest component

    Returns
    -------
    cleaned : (H, W) uint8 {0, 1}
    """
    if mask.sum() == 0 or (min_area_ratio <= 0 and not keep_largest):
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n_labels <= 1:
        return mask

    areas    = stats[1:, cv2.CC_STAT_AREA]        # index 0 is the background
    min_area = min_area_ratio * mask.size

    if keep_largest:
        keep = np.array([int(areas.argmax()) + 1])
    else:
        keep = np.flatnonzero(areas >= min_area) + 1
        if keep.size == 0:                        # everything filtered out
            keep = np.array([int(areas.argmax()) + 1])

    return np.isin(labels, keep).astype(np.uint8)


def _colour_trimap(trimap: np.ndarray) -> np.ndarray:
    vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
    vis[trimap == Label.BG_DEFINITE] = [  0,   0,   0]   # black
    vis[trimap == Label.FG_DEFINITE] = [255, 255, 255]   # white
    vis[trimap == Label.BG_PROBABLE] = [ 60,  20,  20]   # dark red
    vis[trimap == Label.FG_PROBABLE] = [  0, 200, 200]   # cyan
    return vis


class GCNGrabCutPipeline:
    """
    Full GCN-GrabCut segmentation pipeline.

    Parameters
    ----------
    model     : trained trimap predictor (ResGCNNet / GATTrimapNet / GCNTrimapNet)
    sp_config : SuperpixelGraphConfig (uses default 300 segments if None)
    gc_config : GrabCutConfig (uses default 5 iterations if None)
    device    : "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        model,
        sp_config: Optional[SuperpixelGraphConfig] = None,
        gc_config: Optional[GrabCutConfig]         = None,
        device:    str = "cpu",
    ):
        self.model     = model.to(device)
        self.device    = device
        self.sp_config = sp_config or SuperpixelGraphConfig()
        self.gc_config = gc_config or GrabCutConfig()

    # GCN-guided, fully automatic

    def segment(
        self,
        image:          np.ndarray,
        threshold_fg:   float = 0.55,
        threshold_bg:   float = 0.55,
        refine_iters:   int   = 0,
        min_area_ratio: float = 0.002,
        keep_largest:   bool  = False,
        edge_aware:     bool  = True,
        filter_radius:  int   = 8,
    ) -> SegmentationResult:
        """
        Full GCN-GrabCut pipeline — takes an image, returns a mask.

        Parameters
        ----------
        image           : BGR image (H, W, 3)
        threshold_fg/bg : softmax probability thresholds for definite labels
        refine_iters    : extra GrabCut refinement iterations after initial run
        min_area_ratio  : drop connected components smaller than this fraction
                          of the image (0 disables clean-up)
        keep_largest    : keep only the largest connected component
        edge_aware      : project region probabilities to pixels through a
                          guided filter, so trimap boundaries follow image
                          edges instead of superpixel edges
        filter_radius   : guided-filter radius in pixels
        """
        import torch
        from torch_geometric.data import Data as PyGData
        timing: dict[str, float] = {}

        # Build superpixel graph (the automatic FG/BG prior is computed here)
        t = time.perf_counter()
        builder = GraphBuilder(image, self.sp_config)
        graph   = builder.build()
        timing["graph_build"] = time.perf_counter() - t

        # Build PyG data — node input already carries the automatic prior
        t = time.perf_counter()
        data = PyGData(
            x=torch.tensor(graph.node_input(), dtype=torch.float32),
            edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(graph.edge_attr,  dtype=torch.float32),
        ).to(self.device)
        timing["data_prep"] = time.perf_counter() - t

        t = time.perf_counter()
        if edge_aware and hasattr(self.model, "predict_probs"):
            probs  = self.model.predict_probs(data)
            trimap = refine_trimap(
                probs, graph.segments, image,
                threshold_fg, threshold_bg, radius=filter_radius,
            )
        else:
            trimap = self.model.predict_trimap(
                data, graph.segments, threshold_fg, threshold_bg
            )
        timing["gcn_inference"] = time.perf_counter() - t

        # Without a user to correct it, a one-sided trimap must be repaired
        trimap = _seed_from_prior(trimap, graph)

        # GrabCut refinement
        t  = time.perf_counter()
        gc = GrabCut(image, self.gc_config)
        binary_mask = gc.run_with_trimap(trimap)
        if refine_iters > 0:
            binary_mask = gc.refine(refine_iters)
        timing["grabcut"] = time.perf_counter() - t

        # Clean-up — automatic masks have no user to prune stray blobs
        t = time.perf_counter()
        cleaned = clean_mask(binary_mask, min_area_ratio, keep_largest)
        if not np.array_equal(cleaned, binary_mask):
            binary_mask = cleaned
            gc.mask = np.where(binary_mask == 1, Label.FG_PROBABLE,
                               Label.BG_PROBABLE).astype(np.uint8)
        timing["postprocess"] = time.perf_counter() - t

        return SegmentationResult(
            image=image,
            binary_mask=binary_mask,
            trimap=trimap,
            segments=graph.segments,
            overlay=gc.overlay_mask(),
            rgba=gc.crop_foreground(),
            timing=timing,
        )

    def segment_bbox(
        self,
        image: np.ndarray,
        bbox:  tuple[int, int, int, int],
    ) -> SegmentationResult:
        """Classical GrabCut with bounding box — useful as a baseline."""
        gc  = GrabCut(image, self.gc_config)
        binary_mask = gc.run_with_bbox(bbox)

        x, y, w, h = bbox
        H, W = image.shape[:2]
        trimap = np.full((H, W), Label.BG_PROBABLE, dtype=np.uint8)
        trimap[y:y+h, x:x+w] = Label.FG_PROBABLE
        kernel = np.ones((30, 30), np.uint8)
        inner  = np.zeros((H, W), dtype=np.uint8)
        inner[y:y+h, x:x+w] = 1
        inner  = cv2.erode(inner, kernel)
        trimap[inner == 1] = Label.FG_DEFINITE

        return SegmentationResult(
            image=image,
            binary_mask=binary_mask,
            trimap=trimap,
            segments=np.zeros((H, W), dtype=np.int32),
            overlay=gc.overlay_mask(),
            rgba=gc.crop_foreground(),
        )
