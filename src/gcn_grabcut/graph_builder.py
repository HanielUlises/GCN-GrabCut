"""
Superpixel-based graph construction for GCN-GrabCut.

Graph structure
---------------
Nodes  = SLIC superpixels (one node per region)
Edges  = adjacency between spatially neighbouring superpixels (4/8-connectivity)

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

Edge feature vector (4 dims)
-----------------------------
  [0]    colour dissimilarity (ΔE in LAB)
  [1]    spatial distance between centroids (normalised)
  [2]    shared boundary length (normalised)
  [3]    gradient contrast at shared boundary

User hint features (3 dims concatenated at inference/train time)
------------------------------------------------------------------
  [0]    has_fg_click
  [1]    has_bg_click
  [2]    is_unknown (1 if neither clicked)

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

N_IMAGE_FEATS = 16   # Dimensionality of image-derived node features
N_HINT_FEATS  = 3    # User click features
N_NODE_FEATS  = N_IMAGE_FEATS + N_HINT_FEATS
N_EDGE_FEATS  = 4


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

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        nx.set_node_attributes(G, {i: self.node_features[i] for i in range(self.n_nodes)}, "feat")
        for i in range(self.edge_index.shape[1]):
            s, d = self.edge_index[0, i], self.edge_index[1, i]
            if s < d:
                G.add_edge(int(s), int(d), attr=self.edge_attr[i])
        return G

    def to_pyg(self, hint_features: np.ndarray | None = None) -> "PyGData":
        """
        Convert to PyTorch Geometric Data.

        Parameters
        ----------
        hint_features : (N, 3) float32 — optional user hint features.
            If None, all-unknown hints are appended automatically.
        """
        assert _TORCH_AVAILABLE, "torch + torch_geometric required"
        N = self.n_nodes
        if hint_features is None:
            hint_features = np.zeros((N, N_HINT_FEATS), dtype=np.float32)
            hint_features[:, 2] = 1.0   # all unknown
        x = np.concatenate([self.node_features, hint_features], axis=1)   # (N, 19)
        return PyGData(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(self.edge_index, dtype=torch.long),
            edge_attr=torch.tensor(self.edge_attr, dtype=torch.float32),
        )

class GraphBuilder:
    """
    Builds a rich superpixel adjacency graph from a BGR image.

    Example
    -------
    builder = GraphBuilder(image)
    graph   = builder.build()
    pyg     = graph.to_pyg(hint_features)
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
        node_features, centroids = self._compute_node_features(segments)
        edge_index, edge_attr = self._compute_edges(segments)
        n_nodes = int(segments.max()) + 1

        return SuperpixelGraph(
            segments=segments,
            node_features=node_features.astype(np.float32),
            edge_index=edge_index.astype(np.int64),
            edge_attr=edge_attr.astype(np.float32),
            n_nodes=n_nodes,
            n_edges=edge_index.shape[1],
            node_centroids=centroids,
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

    def _compute_node_features(
        self, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        H, W    = segments.shape
        n_nodes = int(segments.max()) + 1
        total_px = H * W

        feats     = np.zeros((n_nodes, N_IMAGE_FEATS), dtype=np.float32)
        centroids = np.zeros((n_nodes, 2),             dtype=np.float32)

        # Boundary map (1 = boundary pixel)
        boundaries = find_boundaries(segments, mode="inner").astype(np.float32)

        for nid in range(n_nodes):
            mask = segments == nid
            n_px = mask.sum()
            if n_px == 0:
                continue

            lab_px = self._lab[mask]      # (k, 3)
            hsv_px = self._hsv[mask]      # (k, 3)
            feats[nid, 0:3] = lab_px.mean(axis=0)
            feats[nid, 3:6] = lab_px.std(axis=0)
            feats[nid, 6:9] = hsv_px.mean(axis=0)

            ys, xs       = np.where(mask)
            cy, cx       = ys.mean() / H, xs.mean() / W
            feats[nid, 9]  = cy
            feats[nid, 10] = cx
            centroids[nid] = [cy, cx]

            feats[nid, 11] = n_px / total_px

            boundary_px = boundaries[mask].sum()
            perimeter   = max(boundary_px, 1.0)
            feats[nid, 12] = (4 * np.pi * n_px) / (perimeter ** 2)

            feats[nid, 13] = self._grad[mask].mean() / 255.0

            feats[nid, 14] = boundary_px / n_px

            feats[nid, 15] = np.sqrt((cy - 0.5)**2 + (cx - 0.5)**2) / 0.707

        for col_range in [slice(0, 3), slice(3, 6)]:
            col = feats[:, col_range]
            mn, mx = col.min(0), col.max(0)
            feats[:, col_range] = (col - mn) / (mx - mn + 1e-6)

        return feats, centroids

    def _compute_edges(
        self, segments: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find adjacent superpixels and compute 4-dim edge features.
        """
        # Adjacent pairs
        pairs_h = np.stack([segments[:, :-1].ravel(), segments[:, 1:].ravel()], 1)
        pairs_v = np.stack([segments[:-1, :].ravel(), segments[1:, :].ravel()], 1)
        if self.config.connectivity == 8:
            pairs_d1 = np.stack([segments[:-1, :-1].ravel(), segments[1:, 1:].ravel()], 1)
            pairs_d2 = np.stack([segments[:-1,  1:].ravel(), segments[1:, :-1].ravel()], 1)
            all_pairs = np.concatenate([pairs_h, pairs_v, pairs_d1, pairs_d2], 0)
        else:
            all_pairs = np.concatenate([pairs_h, pairs_v], 0)

        
        diff_mask  = all_pairs[:, 0] != all_pairs[:, 1]
        all_pairs  = all_pairs[diff_mask]
        sorted_p   = np.sort(all_pairs, axis=1)
        unique_e   = np.unique(sorted_p, axis=0)
        n_undir    = unique_e.shape[0]

        # General stats of nodes
        n_nodes    = int(segments.max()) + 1
        node_means = np.zeros((n_nodes, 3), dtype=np.float32)
        for nid in range(n_nodes):
            m = segments == nid
            if m.any():
                node_means[nid] = self._lab[m].mean(0)

        centroids  = np.zeros((n_nodes, 2), dtype=np.float32)
        H, W       = segments.shape
        for nid in range(n_nodes):
            m = segments == nid
            if m.any():
                ys, xs = np.where(m)
                centroids[nid] = [ys.mean() / H, xs.mean() / W]

        # Colour dissimilarity (ΔE)
        delta_e = np.linalg.norm(
            node_means[unique_e[:, 0]] - node_means[unique_e[:, 1]], axis=1
        )
        delta_e /= delta_e.max() + 1e-6

        # Spatial distance between centroids
        dxy = np.linalg.norm(
            centroids[unique_e[:, 0]] - centroids[unique_e[:, 1]], axis=1
        )
        dxy /= dxy.max() + 1e-6

        # Shared boundary length, boundary pixels that touch both superpixels
        boundaries = find_boundaries(segments, mode="inner")
        shared = np.zeros(n_undir, dtype=np.float32)

        edge_lookup = {(int(unique_e[i, 0]), int(unique_e[i, 1])): i for i in range(n_undir)}

        H_s, W_s = segments.shape
        ys, xs   = np.where(boundaries)
        for y, x in zip(ys, xs):
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx_ = y+dy, x+dx
                if 0 <= ny < H_s and 0 <= nx_ < W_s:
                    a, b = int(segments[y, x]), int(segments[ny, nx_])
                    if a != b:
                        key = (min(a,b), max(a,b))
                        if key in edge_lookup:
                            shared[edge_lookup[key]] += 1

        shared /= shared.max() + 1e-6

    
        grad_contrast = np.zeros(n_undir, dtype=np.float32)
        grad_norm = self._grad / (self._grad.max() + 1e-6)
        for i in range(n_undir):
            a, b = unique_e[i]
            g_a = grad_norm[segments == a].mean() if (segments == a).any() else 0.0
            g_b = grad_norm[segments == b].mean() if (segments == b).any() else 0.0
            grad_contrast[i] = abs(float(g_a) - float(g_b))

        edge_attr_undir = np.stack([delta_e, dxy, shared, grad_contrast], axis=1)

        src = np.concatenate([unique_e[:, 0], unique_e[:, 1]])
        dst = np.concatenate([unique_e[:, 1], unique_e[:, 0]])
        edge_index = np.stack([src, dst], axis=0)
        edge_attr  = np.concatenate([edge_attr_undir, edge_attr_undir], axis=0)

        return edge_index, edge_attr


    def visualize(self, segments: np.ndarray) -> np.ndarray:
        img = mark_boundaries(self.rgb, segments, color=(1, 0.3, 0))
        return (img * 255).astype(np.uint8)


def encode_user_hints(
    segments: np.ndarray,
    fg_points: list[tuple[int, int]],
    bg_points: list[tuple[int, int]],
) -> np.ndarray:
    """
    Build per-superpixel hint feature vector from user clicks.

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
