"""
GCN-GrabCut End-to-End Pipeline.

Orchestrates:
  1. Superpixel graph construction
  2. User hint encoding
  3. GCN inference → trimap
  4. GrabCut refinement → binary mask

Usage
-----
pipeline = GCNGrabCutPipeline(model)
result   = pipeline.segment(image, fg_points=[(r,c),...], bg_points=[(r,c),...])
result.show()
result.save("output")
"""

from __future__ import annotations

import numpy as np
import cv2
import time
from dataclasses import dataclass, field
from typing import Optional

from .grabcut import GrabCut, GrabCutConfig, Label
from .graph_builder import GraphBuilder, SuperpixelGraphConfig, encode_user_hints
from .metrics import evaluate, evaluate_trimap, SegmentationMetrics, TrimapMetrics


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

    # GCN-guided

    def segment(
        self,
        image:        np.ndarray,
        fg_points:    list[tuple[int, int]],
        bg_points:    list[tuple[int, int]],
        threshold_fg: float = 0.55,
        threshold_bg: float = 0.55,
        refine_iters: int   = 0,
    ) -> SegmentationResult:
        """
        Full GCN-GrabCut pipeline.

        Parameters
        ----------
        image      : BGR image (H, W, 3)
        fg_points  : (row, col) foreground user clicks
        bg_points  : (row, col) background user clicks
        threshold_fg/bg : softmax probability thresholds for definite labels
        refine_iters    : extra GrabCut refinement iterations after initial run
        """
        import torch
        from torch_geometric.data import Data as PyGData
        timing: dict[str, float] = {}

        # Build superpixel graph
        t = time.perf_counter()
        builder = GraphBuilder(image, self.sp_config)
        graph   = builder.build()
        timing["graph_build"] = time.perf_counter() - t

        # Encode hints + build PyG data
        t = time.perf_counter()
        hints = encode_user_hints(graph.segments, fg_points, bg_points)
        x     = np.concatenate([graph.node_features, hints], axis=1)
        data  = PyGData(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(graph.edge_attr,  dtype=torch.float32),
        ).to(self.device)
        timing["data_prep"] = time.perf_counter() - t

        # GCN inference, we get the trimap
        t      = time.perf_counter()
        trimap = self.model.predict_trimap(
            data, graph.segments, threshold_fg, threshold_bg
        )
        timing["gcn_inference"] = time.perf_counter() - t

        # GrabCut refinement
        t  = time.perf_counter()
        gc = GrabCut(image, self.gc_config)
        binary_mask = gc.run_with_trimap(trimap)
        if refine_iters > 0:
            binary_mask = gc.refine(refine_iters)
        timing["grabcut"] = time.perf_counter() - t

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
