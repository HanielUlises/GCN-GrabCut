"""
Superpixel-based graph construction for GCN-GrabCut.

Graph structure
---------------
Nodes  = SLIC superpixels (one node per region)
Edges  = adjacency between spatially neighbouring superpixels (4/8-connectivity)
         plus optional non-local edges linking colour-similar regions that are
         not spatially adjacent (k nearest neighbours in mean-LAB space)

Node feature vector (16 dims)
------------------------------
  [0:3]   mean LAB colour
  [3:6]   std  LAB colour
  [6:9]   mean HSV (hue/saturation/value statistics)
  [9]     centroid Y (normalised)
  [10]    centroid X (normalised)
  [11]    size ratio (pixels / total pixels)
  [12]    compactness (circularity measure)
  [13]    mean gradient magnitude inside superpixel
  [14]    boundary pixel ratio (fraction of pixels on superpixel boundary)
  [15]    distance to image centre (normalised)

Edge feature vector (5 dims)
-----------------------------
  [0]    colour dissimilarity (ΔE in LAB)
  [1]    spatial distance between centroids (normalised)
  [2]    shared boundary length (normalised; 0 for non-local edges)
  [3]    gradient contrast between the two regions
  [4]    edge type (0 = spatial adjacency, 1 = non-local colour edge)

Automatic prior features (3 dims, computed from the image itself)
------------------------------------------------------------------
  [0]    foreground-ness  (global colour contrast × centre prior)
  [1]    background-ness  (image-border colour model / boundary connectivity)
  [2]    ambiguity        (1 - |fg-ness - bg-ness|)

These replace the old user-click channels: the pipeline is fully automatic
and never asks the user to touch the image.

Total node features fed to GCN: 16 + 3 = 19
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple

from skimage.segmentation import slic, find_boundaries, mark_boundaries
from skimage.color import rgb2lab, rgb2hsv
import networkx as nx

try:
    import torch
    from torch_geometric.data import Data as PyGData
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False



@dataclass
class SuperpixelGraphConfig:
    n_segments:  int   = 300    # Target superpixel count
    compactness: float = 10.0   # SLIC spatial regularisation (higher = more square)
    sigma:       float = 1.0    # Gaussian pre-smoothing
    use_lab:     bool  = True   # Use LAB for SLIC (better perceptual grouping)
    connectivity: int = 4       # 4 or 8 — pixel adjacency for edge detection
    n_nonlocal:  int   = 4      # non-local colour neighbours per node (0 = off)

N_IMAGE_FEATS = 16   # Dimensionality of image-derived node features
N_PRIOR_FEATS = 3    # Automatic FG/BG/ambiguity prior
N_HINT_FEATS  = N_PRIOR_FEATS   # Backwards-compatible alias
N_NODE_FEATS  = N_IMAGE_FEATS + N_PRIOR_FEATS
N_EDGE_FEATS  = 5


@dataclass
class SuperpixelGraph:
    """Container for a built superpixel graph."""
    segments:       np.ndarray   # (H, W) int32 superpixel ID per pixel
    node_features:  np.ndarray   # (N, N_IMAGE_FEATS) float32
    edge_index:     np.ndarray   # (2, E) int64 COO format (bidirectional)
    edge_attr:      np.ndarray   # (E, N_EDGE_FEATS) float32
    n_nodes:        int = 0
    n_edges:        int = 0
    node_centroids: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    prior_features: np.ndarray = field(default_factory=lambda: np.empty((0, N_PRIOR_FEATS)))
    node_areas:     np.ndarray = field(default_factory=lambda: np.empty((0,)))

    def node_input(self, prior_features: np.ndarray | None = None) -> np.ndarray:
        """Full (N, 19) node input matrix = image features ‖ automatic prior."""
        prior = self.prior_features if prior_features is None else prior_features
        if prior is None or prior.size == 0:
            prior = np.zeros((self.n_nodes, N_PRIOR_FEATS), dtype=np.float32)
        return np.concatenate([self.node_features, prior], axis=1).astype(np.float32)

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        nx.set_node_attributes(G, {i: self.node_features[i] for i in range(self.n_nodes)}, "feat")
        for i in range(self.edge_index.shape[1]):
            s, d = self.edge_index[0, i], self.edge_index[1, i]
            if s < d:
                G.add_edge(int(s), int(d), attr=self.edge_attr[i])
        return G

    def to_pyg(self, prior_features: np.ndarray | None = None) -> "PyGData":
        """
        Convert to PyTorch Geometric Data.

        Parameters
        ----------
        prior_features : (N, 3) float32 — optional override for the automatic
            prior. If None, the prior computed during `build()` is used.
        """
        assert _TORCH_AVAILABLE, "torch + torch_geometric required"
        x = self.node_input(prior_features)   # (N, 19)
        area = self.node_areas
        if area is None or area.size == 0:
            area = np.full(self.n_nodes, 1.0 / max(self.n_nodes, 1), dtype=np.float32)
        return PyGData(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float32),
            node_area=torch.tensor(area, dtype=torch.float32),
        )

class GraphBuilder:
    """
    Builds a rich superpixel adjacency graph from a BGR image.

    Example
    -------
    builder = GraphBuilder(image)
    graph   = builder.build()      # node features + automatic FG/BG prior
    pyg     = graph.to_pyg()
    """

    def __init__(self, image: np.ndarray, config: SuperpixelGraphConfig | None = None):
        """image : BGR uint8 (H, W, 3)"""
        self.bgr    = image
        self.rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.config = config or SuperpixelGraphConfig()
        
        self._lab  = rgb2lab(self.rgb).astype(np.float32)
        self._hsv  = rgb2hsv(self.rgb).astype(np.float32)
        self._gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

        gx = cv2.Sobel(self._gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(self._gray, cv2.CV_32F, 0, 1, ksize=3)
        self._grad = np.sqrt(gx**2 + gy**2)

    def build(self) -> SuperpixelGraph:
        segments = self._compute_superpixels()
        n_nodes  = int(segments.max()) + 1

        stats = self._region_statistics(segments, n_nodes)
        node_features = self._assemble_node_features(segments, stats)
        edge_index, edge_attr = self._compute_edges(segments, stats)
        prior = compute_auto_prior(segments, self._lab)

        return SuperpixelGraph(
            segments=segments,
            node_features=node_features.astype(np.float32),
            edge_index=edge_index.astype(np.int64),
            edge_attr=edge_attr.astype(np.float32),
            n_nodes=n_nodes,
            n_edges=edge_index.shape[1],
            node_centroids=stats["centroids"],
            prior_features=prior,
            node_areas=stats["area_ratio"],
        )

    def _compute_superpixels(self) -> np.ndarray:
        cfg = self.config
        img = self._lab if cfg.use_lab else self.rgb.astype(float)
        segments = slic(
            img,
            n_segments=cfg.n_segments,
            compactness=cfg.compactness,
            sigma=cfg.sigma,
            start_label=0,
            channel_axis=-1,
        )
        return segments.astype(np.int32)

    def _region_statistics(self, segments: np.ndarray, n_nodes: int) -> dict:
        H, W = segments.shape
        flat = segments.ravel()

        counts = np.bincount(flat, minlength=n_nodes).astype(np.float32)
        safe   = np.maximum(counts, 1.0)

        def _sum(field: np.ndarray) -> np.ndarray:
            return np.bincount(flat, weights=field.ravel(),
                               minlength=n_nodes).astype(np.float32)

        mean_lab = np.stack([_sum(self._lab[:, :, c]) for c in range(3)], 1) / safe[:, None]
        sq_lab   = np.stack([_sum(self._lab[:, :, c] ** 2) for c in range(3)], 1) / safe[:, None]
        std_lab  = np.sqrt(np.maximum(sq_lab - mean_lab ** 2, 0.0))
        mean_hsv = np.stack([_sum(self._hsv[:, :, c]) for c in range(3)], 1) / safe[:, None]

        yy, xx    = np.mgrid[0:H, 0:W]
        cy        = _sum(yy.astype(np.float32) / H) / safe
        cx        = _sum(xx.astype(np.float32) / W) / safe
        centroids = np.stack([cy, cx], 1).astype(np.float32)

        boundaries  = find_boundaries(segments, mode="inner").astype(np.float32)
        boundary_px = _sum(boundaries)

        grad_scaled = self._grad / (self._grad.max() + 1e-6)
        return {
            "counts":      counts,
            "safe":        safe,
            "area_ratio":  (counts / float(H * W)).astype(np.float32),
            "mean_lab":    mean_lab.astype(np.float32),
            "std_lab":     std_lab.astype(np.float32),
            "mean_hsv":    mean_hsv.astype(np.float32),
            "centroids":   centroids,
            "boundary_px": boundary_px,
            "mean_grad":   (_sum(self._grad) / safe).astype(np.float32),
            "mean_grad_n": (_sum(grad_scaled) / safe).astype(np.float32),
        }

    def _assemble_node_features(self, segments: np.ndarray, st: dict) -> np.ndarray:
        n_nodes = st["counts"].shape[0]
        feats   = np.zeros((n_nodes, N_IMAGE_FEATS), dtype=np.float32)

        feats[:, 0:3]  = st["mean_lab"]
        feats[:, 3:6]  = st["std_lab"]
        feats[:, 6:9]  = st["mean_hsv"]
        feats[:, 9]    = st["centroids"][:, 0]
        feats[:, 10]   = st["centroids"][:, 1]
        feats[:, 11]   = st["area_ratio"]

        # Isoperimetric ratio, clipped to its theoretical range: the inner
        # boundary undercounts the perimeter of very small regions, which
        # otherwise produces feature values orders of magnitude off scale.
        perimeter      = np.maximum(st["boundary_px"], 1.0)
        feats[:, 12]   = np.clip((4 * np.pi * st["counts"]) / (perimeter ** 2), 0.0, 1.0)
        feats[:, 13]   = st["mean_grad"] / 255.0
        feats[:, 14]   = st["boundary_px"] / st["safe"]
        feats[:, 15]   = np.linalg.norm(st["centroids"] - 0.5, axis=1) / 0.707

        # Colour statistics are min-max normalised per image so that the
        # network sees a comparable input range regardless of exposure.
        for col_range in (slice(0, 3), slice(3, 6)):
            col = feats[:, col_range]
            mn, mx = col.min(0), col.max(0)
            feats[:, col_range] = (col - mn) / (mx - mn + 1e-6)

        return np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=0.0)

    def _compute_edges(self, segments: np.ndarray, st: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Region-adjacency edges (plus optional non-local colour edges).

        The number of adjacent pixel pairs between two regions is exactly the
        shared boundary length, so counting unique pairs yields both the edge
        set and the boundary-length feature in one vectorised operation.
        """
        n_nodes = st["counts"].shape[0]

        shifts = [
            (segments[:, :-1], segments[:, 1:]),
            (segments[:-1, :], segments[1:, :]),
        ]
        if self.config.connectivity == 8:
            shifts += [
                (segments[:-1, :-1], segments[1:, 1:]),
                (segments[:-1, 1:],  segments[1:, :-1]),
            ]

        a = np.concatenate([s[0].ravel() for s in shifts])
        b = np.concatenate([s[1].ravel() for s in shifts])
        keep = a != b
        a, b = a[keep], b[keep]
        lo, hi = np.minimum(a, b), np.maximum(a, b)

        codes, shared = np.unique(lo.astype(np.int64) * n_nodes + hi.astype(np.int64),
                                  return_counts=True)
        pairs  = np.stack([codes // n_nodes, codes % n_nodes], 1)
        shared = shared.astype(np.float32) / (shared.max() + 1e-6)

        attr = self._pair_features(pairs, st, shared,
                                   nonlocal_flag=np.zeros(len(pairs), np.float32))

        if self.config.n_nonlocal > 0 and n_nodes > self.config.n_nonlocal + 1:
            nl_pairs = self._nonlocal_pairs(pairs, st, n_nodes)
            if len(nl_pairs):
                nl_attr = self._pair_features(
                    nl_pairs, st,
                    np.zeros(len(nl_pairs), np.float32),
                    nonlocal_flag=np.ones(len(nl_pairs), np.float32),
                )
                pairs = np.concatenate([pairs, nl_pairs], 0)
                attr  = np.concatenate([attr, nl_attr], 0)

        # Undirected graph stored as symmetric directed pairs
        src = np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
        edge_index = np.stack([src, dst], 0)
        edge_attr  = np.concatenate([attr, attr], 0)
        return edge_index, edge_attr

    def _pair_features(self, pairs, st, shared, nonlocal_flag) -> np.ndarray:
        """Assemble the 5-dim feature vector for a set of undirected pairs."""
        i, j = pairs[:, 0], pairs[:, 1]

        delta_e = np.linalg.norm(st["mean_lab"][i] - st["mean_lab"][j], axis=1)
        delta_e = delta_e / (delta_e.max() + 1e-6)

        dxy = np.linalg.norm(st["centroids"][i] - st["centroids"][j], axis=1)
        dxy = dxy / (dxy.max() + 1e-6)

        grad_contrast = np.abs(st["mean_grad_n"][i] - st["mean_grad_n"][j])

        return np.stack([delta_e, dxy, shared, grad_contrast, nonlocal_flag],
                        axis=1).astype(np.float32)

    def _nonlocal_pairs(self, adj_pairs, st, n_nodes) -> np.ndarray:
        """
        Link every region to its k nearest neighbours in mean-LAB space,
        excluding pairs that are already spatially adjacent.

        These edges give message passing a non-local path, so evidence about
        an object can travel between disconnected parts of the same object
        (e.g. a limb separated by an occluder) without stacking extra layers.
        """
        k = int(self.config.n_nonlocal)
        d = np.linalg.norm(st["mean_lab"][:, None, :] - st["mean_lab"][None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)

        adj = np.zeros((n_nodes, n_nodes), dtype=bool)
        adj[adj_pairs[:, 0], adj_pairs[:, 1]] = True
        adj[adj_pairs[:, 1], adj_pairs[:, 0]] = True
        d[adj] = np.inf

        nbrs = np.argpartition(d, kth=min(k, n_nodes - 1) - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(n_nodes), k)
        cols = nbrs.ravel()

        valid = np.isfinite(d[rows, cols])
        rows, cols = rows[valid], cols[valid]
        lo, hi = np.minimum(rows, cols), np.maximum(rows, cols)
        codes  = np.unique(lo.astype(np.int64) * n_nodes + hi.astype(np.int64))
        return np.stack([codes // n_nodes, codes % n_nodes], 1)

    def visualize(self, segments: np.ndarray) -> np.ndarray:
        img = mark_boundaries(self.rgb, segments, color=(1, 0.3, 0))
        return (img * 255).astype(np.uint8)


def compute_auto_prior(
    segments:      np.ndarray,
    lab:           np.ndarray,
    centre_sigma:  float = 0.45,
    contrast_sigma: float = 0.40,
) -> np.ndarray:
    """
    Compute a per-superpixel foreground/background prior from the image alone.

    This is the automatic replacement for user clicks. It combines two classical,
    training-free saliency cues:

    * **Global colour contrast** (Cheng et al., "Global Contrast based Salient
      Region Detection") — a region is salient when its colour differs from the
      rest of the image, weighted by region area and spatial proximity.
    * **Boundary connectivity** (Zhu et al., "Saliency Optimization from Robust
      Background Detection") — regions touching the image frame are background
      seeds; every region is scored against the resulting border colour model.

    Parameters
    ----------
    segments : (H, W) int32 superpixel map
    lab      : (H, W, 3) float32 CIELAB image
    centre_sigma   : width of the Gaussian centre prior (normalised units)
    contrast_sigma : spatial falloff for the contrast sum (normalised units)

    Returns
    -------
    prior : (N, 3) float32 — columns [fg-ness, bg-ness, ambiguity], each in [0, 1]
    """
    H, W    = segments.shape
    n_nodes = int(segments.max()) + 1
    flat    = segments.ravel()

    counts = np.bincount(flat, minlength=n_nodes).astype(np.float32)
    safe   = np.maximum(counts, 1.0)

    # Per-node mean LAB colour and centroid (vectorised)
    mean_lab = np.stack(
        [np.bincount(flat, weights=lab[:, :, c].ravel(), minlength=n_nodes)
         for c in range(3)], axis=1
    ).astype(np.float32) / safe[:, None]

    yy, xx = np.mgrid[0:H, 0:W]
    cy = np.bincount(flat, weights=(yy.ravel() / H), minlength=n_nodes) / safe
    cx = np.bincount(flat, weights=(xx.ravel() / W), minlength=n_nodes) / safe
    centroids = np.stack([cy, cx], axis=1).astype(np.float32)

    # --- Cue 1: spatially weighted global colour contrast -------------------
    colour_d = np.linalg.norm(mean_lab[:, None, :] - mean_lab[None, :, :], axis=2)
    spatial_d = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
    spatial_w = np.exp(-(spatial_d ** 2) / (2 * contrast_sigma ** 2))
    area_w    = counts / max(counts.sum(), 1.0)

    contrast = (colour_d * spatial_w * area_w[None, :]).sum(axis=1)
    contrast = _unit_norm(contrast)

    # Centre prior — salient objects rarely sit in the extreme corners
    centre_d = np.linalg.norm(centroids - 0.5, axis=1)
    centre_w = np.exp(-(centre_d ** 2) / (2 * centre_sigma ** 2))

    fgness = _unit_norm(contrast * centre_w)

    # --- Cue 2: background model from image-border superpixels --------------
    border_ids = np.concatenate([
        segments[0, :], segments[-1, :], segments[:, 0], segments[:, -1]
    ])
    border_count = np.bincount(border_ids, minlength=n_nodes).astype(np.float32)
    border_ratio = border_count / safe          # fraction of node pixels on the frame

    if border_count.sum() > 0:
        w_bg  = border_count / border_count.sum()
        mu_bg = (mean_lab * w_bg[:, None]).sum(axis=0)
        var_bg = (((mean_lab - mu_bg) ** 2) * w_bg[:, None]).sum(axis=0).sum()
        sigma_bg = float(np.sqrt(max(var_bg, 1e-6)))
        d_bg  = np.linalg.norm(mean_lab - mu_bg, axis=1)
        bgness = np.exp(-(d_bg ** 2) / (2 * (sigma_bg + 1e-6) ** 2))
    else:                                        # degenerate single-node image
        bgness = np.zeros(n_nodes, dtype=np.float32)

    # Touching the frame is direct evidence of background
    bgness = _unit_norm(np.maximum(bgness, np.clip(border_ratio * 4.0, 0.0, 1.0)))

    # --- Ambiguity: high where the two cues disagree or are both weak -------
    ambiguity = 1.0 - np.abs(fgness - bgness)

    prior = np.stack([fgness, bgness, ambiguity], axis=1).astype(np.float32)
    return np.nan_to_num(prior, nan=0.0, posinf=1.0, neginf=0.0)


def _unit_norm(v: np.ndarray) -> np.ndarray:
    """Scale a vector to [0, 1]; constant vectors map to zeros."""
    v  = v.astype(np.float32)
    mn = float(v.min())
    mx = float(v.max())
    if mx - mn < 1e-8:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def encode_user_hints(
    segments: np.ndarray,
    fg_points: list[tuple[int, int]],
    bg_points: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build per-superpixel hint features from explicit clicks.

    Legacy/optional: the pipeline is fully automatic and does not call this.
    Kept so a caller can still override the automatic prior with hard
    constraints (e.g. for ablation studies against the interactive baseline).

    Returns
    -------
    hints : (N, 3) float32
        Column 0: superpixel received ≥1 FG click
        Column 1: superpixel received ≥1 BG click
        Column 2: superpixel is 'unknown' (neither clicked)
    """
    n_nodes = int(segments.max()) + 1
    hints   = np.zeros((n_nodes, 3), dtype=np.float32)
    hints[:, 2] = 1.0

    for r, c in fg_points:
        r, c = int(r), int(c)
        if 0 <= r < segments.shape[0] and 0 <= c < segments.shape[1]:
            nid = int(segments[r, c])
            hints[nid, 0] = 1.0
            hints[nid, 2] = 0.0

    for r, c in bg_points:
        r, c = int(r), int(c)
        if 0 <= r < segments.shape[0] and 0 <= c < segments.shape[1]:
            nid = int(segments[r, c])
            hints[nid, 1] = 1.0
            hints[nid, 2] = 0.0

    return hints
