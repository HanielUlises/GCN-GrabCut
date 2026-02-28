"""
Classical GrabCut wrapper with extensible hooks for GCN-based trimap prediction.

Supports:
  - Bounding-box initialisation (classical mode)
  - Externally predicted trimap (GCN-guided mode)
  - Multiple colour spaces: RGB, HSV, Lab
  - Per-iteration state logging for research

Reference:
  Rother et al. (2004) "GrabCut: Interactive foreground extraction using
  iterated graph cuts", ACM SIGGRAPH.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple, List, Dict


class Label(IntEnum):
    """Pixel label constants — matches OpenCV GrabCut convention."""
    BG_DEFINITE = 0   # cv2.GC_BGD
    FG_DEFINITE = 1   # cv2.GC_FGD
    BG_PROBABLE = 2   # cv2.GC_PR_BGD
    FG_PROBABLE = 3   # cv2.GC_PR_FGD


@dataclass
class GrabCutConfig:
    n_iter:       int   = 5
    n_components: int   = 5       # GMM components per class (OpenCV default)
    gamma:        float = 50.0    # Pairwise smoothness weight
    color_space:  str   = "rgb"   # "rgb" | "hsv" | "lab"


@dataclass
class GrabCutSnapshot:
    """State snapshot captured after each GrabCut run."""
    tag:        str
    fg_pixels:  int
    bg_pixels:  int
    fg_ratio:   float
    mask_copy:  np.ndarray = field(repr=False)


class GrabCut:
    """
    Thin wrapper around cv2.grabCut with logging and visualisation helpers.

    Usage
    -----
    gc = GrabCut(image)
    mask = gc.run_with_bbox((x, y, w, h))    # classical mode
    mask = gc.run_with_trimap(trimap)         # GCN-guided mode
    overlay = gc.overlay_mask()
    rgba    = gc.crop_foreground()
    """

    def __init__(self, image: np.ndarray, config: Optional[GrabCutConfig] = None):
        """image : BGR uint8 (H, W, 3)"""
        self.image  = image
        self.config = config or GrabCutConfig()
        self.mask:  Optional[np.ndarray] = None
        self._bgd   = np.zeros((1, 65), np.float64)
        self._fgd   = np.zeros((1, 65), np.float64)
        self.history: List[GrabCutSnapshot] = []
        self._proc  = self._preprocess(image)

    # ------------------------------------------------------------------  preprocessing

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        cs = self.config.color_space.lower()
        if cs == "hsv":
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if cs == "lab":
            return cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        return image.copy()

    def run_with_bbox(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Classical GrabCut with bounding-box initialisation.

        Parameters
        ----------
        bbox : (x, y, w, h) — top-left corner, width, height

        Returns
        -------
        binary_mask : uint8 array {0=BG, 1=FG}
        """
        self.mask = np.zeros(self.image.shape[:2], np.uint8)
        self._bgd = np.zeros((1, 65), np.float64)
        self._fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            self._proc, self.mask, bbox,
            self._bgd, self._fgd,
            self.config.n_iter, cv2.GC_INIT_WITH_RECT,
        )
        self._snapshot("bbox_init")
        return self._binary()

    def run_with_trimap(self, trimap: np.ndarray) -> np.ndarray:
        """
        GCN-guided GrabCut seeded with a predicted trimap.

        Parameters
        ----------
        trimap : uint8 array (H, W) with values in {0,1,2,3}
            0 = GC_BGD (definite background)
            1 = GC_FGD (definite foreground)
            2 = GC_PR_BGD (probable background)
            3 = GC_PR_FGD (probable foreground)

        Returns
        -------
        binary_mask : uint8 {0, 1}
        """
        if trimap.shape != self.image.shape[:2]:
            raise ValueError(
                f"Trimap shape {trimap.shape} != image shape {self.image.shape[:2]}"
            )
        if trimap.dtype != np.uint8:
            trimap = trimap.astype(np.uint8)

        # GrabCut requires at least one FG and one BG pixel
        if not (trimap == cv2.GC_FGD).any():
            trimap = trimap.copy()
            trimap[trimap == cv2.GC_PR_FGD] = cv2.GC_FGD
        if not (trimap == cv2.GC_BGD).any():
            trimap = trimap.copy()
            trimap[trimap == cv2.GC_PR_BGD] = cv2.GC_BGD

        self.mask = trimap.copy()
        self._bgd = np.zeros((1, 65), np.float64)
        self._fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(
            self._proc, self.mask, None,
            self._bgd, self._fgd,
            self.config.n_iter, cv2.GC_INIT_WITH_MASK,
        )
        self._snapshot("trimap_init")
        return self._binary()

    def refine(self, extra_iter: int = 3) -> np.ndarray:
        """Continue from current GMM state."""
        if self.mask is None:
            raise RuntimeError("Call run_with_bbox or run_with_trimap first.")
        cv2.grabCut(
            self._proc, self.mask, None,
            self._bgd, self._fgd,
            extra_iter, cv2.GC_EVAL,
        )
        self._snapshot("refinement")
        return self._binary()

    def _binary(self) -> np.ndarray:
        return np.where(
            (self.mask == cv2.GC_FGD) | (self.mask == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)

    def _snapshot(self, tag: str) -> None:
        b = self._binary()
        self.history.append(GrabCutSnapshot(
            tag=tag,
            fg_pixels=int(b.sum()),
            bg_pixels=int((b == 0).sum()),
            fg_ratio=float(b.mean()),
            mask_copy=self.mask.copy(),
        ))

    def overlay_mask(self, alpha: float = 0.45, color: Tuple = (0, 220, 100)) -> np.ndarray:
        """Return BGR image with a coloured foreground overlay."""
        binary  = self._binary()
        overlay = self.image.copy().astype(np.float32)
        tint    = np.zeros_like(overlay)
        tint[:] = color[::-1]          # BGR
        mask3   = np.stack([binary] * 3, axis=-1).astype(np.float32)
        overlay = overlay * (1 - alpha * mask3) + tint * alpha * mask3
        return np.clip(overlay, 0, 255).astype(np.uint8)

    def crop_foreground(self) -> np.ndarray:
        """Return BGRA image with background set to transparent."""
        binary = self._binary()
        rgba   = cv2.cvtColor(self.image, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = binary * 255
        return rgba

    def trimap_visualisation(self, trimap: np.ndarray) -> np.ndarray:
        """Render a trimap as a colour image for debugging."""
        vis = np.zeros((*trimap.shape, 3), dtype=np.uint8)
        vis[trimap == Label.BG_DEFINITE] = [  0,   0,   0]   # black
        vis[trimap == Label.FG_DEFINITE] = [255, 255, 255]   # white
        vis[trimap == Label.BG_PROBABLE] = [ 80,   0,   0]   # dark red
        vis[trimap == Label.FG_PROBABLE] = [  0, 200, 200]   # cyan
        return vis
