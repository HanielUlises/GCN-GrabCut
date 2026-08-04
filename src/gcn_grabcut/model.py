"""
GCN Architecture for Trimap Prediction — GCN-GrabCut
=====================================================

Three model variants, increasing in sophistication:

1.  GCNTrimapNet  —  Pure GCNConv stack with residuals + edge-feature injection
2.  GATTrimapNet  —  Multi-head Graph Attention + edge gate
3.  ResGCNNet     —  Deep residual GCN with dense skip connections (best default)

All models share the same interface:
    forward(data)        → logits (N, n_classes)
    predict_trimap(...)  → pixel trimap (H, W) uint8

Architecture:
* Edge features are encoded once into a per-node context vector and reused by
  every layer, instead of being re-projected and re-scattered layer by layer
* Global context is pooled per graph, so several graphs can share a batch
* Jumping-knowledge fusion: layer outputs are combined by learned weights
  rather than concatenated, which keeps the head small at any depth
* Input features are standardised by a running-statistics norm, so raw
  descriptors of very different scale can be fed directly
* Label propagation from an automatic FG/BG prior via prior-conditioned messages
* 3-class output: {BG=0, UNKNOWN=1, FG=2}

Node input: 19-dim  (16 image features + 3 automatic prior features)
Edge input: 5-dim

All models are permutation-equivariant per node and batch-safe: passing a
`torch_geometric.data.Batch` of several graphs gives the same per-node output
as running each graph on its own.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv
    from torch_geometric.data import Data
    _TORCH_GEOMETRIC = True
except ImportError:
    _TORCH_GEOMETRIC = False

from .graph_builder import N_NODE_FEATS, N_EDGE_FEATS, N_PRIOR_FEATS


TRIMAP_BG      = 0   # cv2.GC_BGD
TRIMAP_FG      = 1   # cv2.GC_FGD
TRIMAP_PROB_BG = 2   # cv2.GC_PR_BGD
TRIMAP_PROB_FG = 3   # cv2.GC_PR_FGD

CLASS_BG  = 0
CLASS_UNK = 1
CLASS_FG  = 2


if _TORCH:

    def _scatter_mean(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        """Mean of `src` rows grouped by `index`, over `n` groups."""
        out = torch.zeros(n, src.size(1), device=src.device, dtype=src.dtype)
        out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
        cnt = torch.bincount(index, minlength=n).to(src.dtype).clamp(min=1)
        return out / cnt.unsqueeze(1)


    def _graph_mean(h: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Per-graph mean of node features, broadcast back to every node.

        Reducing over the whole node axis would mix graphs together, which is
        what previously forced training to run one graph at a time.
        """
        if batch is None:
            return h.mean(dim=0, keepdim=True).expand_as(h)
        n_graphs = int(batch.max().item()) + 1
        return _scatter_mean(h, batch, n_graphs)[batch]


    def _graph_softmax(scores: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Softmax over the nodes of each graph independently. `scores` is (N, 1).

        The normalisation is carried out in float32 and cast back: under
        autocast the exponential is evaluated in float32 while the incoming
        scores are half, and the scatter requires both to agree.
        """
        if batch is None:
            return torch.softmax(scores.float(), dim=0).to(scores.dtype)

        n_graphs = int(batch.max().item()) + 1
        s    = scores.float()
        peak = torch.full((n_graphs, 1), float("-inf"),
                          device=s.device, dtype=s.dtype)
        peak = peak.index_reduce(0, batch, s, "amax", include_self=True)
        ex   = torch.exp(s - peak[batch])
        tot  = torch.zeros_like(peak).index_add_(0, batch, ex)
        return (ex / (tot[batch] + 1e-12)).to(scores.dtype)


    class EdgeContext(nn.Module):
        """
        Encode edge features once into a per-node context vector.

        Edge attributes are constant across depth, so projecting and
        scattering them inside every layer repeats identical work. Encoding
        them a single time and letting each layer read the result costs one
        scatter per forward pass instead of one per layer, and removes the
        per-layer edge MLPs from the parameter count.
        """
        def __init__(self, edge_dim: int, hidden_dim: int, ctx_dim: Optional[int] = None):
            super().__init__()
            ctx_dim = ctx_dim or max(hidden_dim // 2, 8)
            self.encode = nn.Sequential(
                nn.Linear(edge_dim, ctx_dim),
                nn.GELU(),
                nn.Linear(ctx_dim, ctx_dim),
            )
            self.to_gate = nn.Sequential(
                nn.LayerNorm(ctx_dim),
                nn.Linear(ctx_dim, hidden_dim),
                nn.Sigmoid(),
            )

        def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor,
                    n_nodes: int) -> torch.Tensor:
            """Returns a multiplicative gate in (0, 1) of shape (n_nodes, hidden_dim)."""
            ctx = _scatter_mean(self.encode(edge_attr), edge_index[1], n_nodes)
            return self.to_gate(ctx)


    class EdgeInjectionLayer(nn.Module):
        """
        Projects edge features and uses them to gate node updates, so messages
        are weighted by colour dissimilarity, spatial distance and so on.

        Used by the baseline and attention variants; `ResGCNNet` uses the
        cheaper shared `EdgeContext` instead.
        """
        def __init__(self, edge_dim: int, hidden_dim: int):
            super().__init__()
            self.proj = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )

        def forward(self, edge_attr: torch.Tensor, edge_index: torch.Tensor, n_nodes: int,
                    node_updates: torch.Tensor) -> torch.Tensor:
            gates = _scatter_mean(self.proj(edge_attr), edge_index[1], n_nodes)
            return node_updates * gates.to(node_updates.dtype)


    class GlobalContextModule(nn.Module):
        """
        Attention-weighted graph readout broadcast back to all nodes.
        Salient and boundary nodes get higher attention weight naturally.
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attn     = nn.Linear(hidden_dim, 1)
            self.compress = nn.Linear(hidden_dim, hidden_dim // 2)
            self.expand   = nn.Linear(hidden_dim // 2, hidden_dim)

        def forward(self, x: torch.Tensor,
                    batch: Optional[torch.Tensor] = None) -> torch.Tensor:
            w = _graph_softmax(self.attn(x), batch).to(x.dtype)
            if batch is None:
                g = (w * x).sum(dim=0, keepdim=True)
            else:
                n_graphs = int(batch.max().item()) + 1
                g = torch.zeros(n_graphs, x.size(1), device=x.device, dtype=x.dtype)
                g = g.index_add_(0, batch, w * x)
                g = g[batch]
            g = F.relu(self.compress(g))
            g = torch.sigmoid(self.expand(g))
            return x * g


    class InputNorm(nn.Module):
        """
        Standardise raw node descriptors with running statistics.

        The 19 input channels are heterogeneous (normalised colours, areas of
        order 1e-3, gradient magnitudes, prior scores). Whitening them removes
        the need to hand-scale features and measurably shortens the number of
        epochs before the loss starts to move.
        """
        def __init__(self, n_features: int, momentum: float = 0.05):
            super().__init__()
            self.norm = nn.BatchNorm1d(n_features, momentum=momentum, affine=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # A single-node graph has no variance to estimate; fall back to
            # the stored statistics rather than dividing by ~0.
            if self.training and x.size(0) < 2:
                was = self.norm.training
                self.norm.eval()
                out = self.norm(x)
                self.norm.train(was)
                return out
            return self.norm(x)


    class ResGCNBlock(nn.Module):
        def __init__(self, in_dim, out_dim, edge_dim, dropout):
            super().__init__()
            self.conv        = GCNConv(in_dim, out_dim)
            self.bn          = nn.BatchNorm1d(out_dim)
            self.dropout     = dropout
            self.skip        = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
            self.edge_inject = EdgeInjectionLayer(edge_dim, out_dim)

        def forward(self, x, edge_index, edge_attr):
            h = self.conv(x, edge_index)
            h = self.bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + self.skip(x)
            h = self.edge_inject(edge_attr, edge_index, x.size(0), h)
            return h


    # -----------------------------------------------------------------------
    # Model 1: GCNTrimapNet
    # -----------------------------------------------------------------------

    class GCNTrimapNet(nn.Module):
        """
        Baseline GCN with residual blocks + edge injection + global context.

        Parameters
        ----------
        in_channels     : input node feature dim (default: N_NODE_FEATS=19)
        edge_channels   : edge feature dim (default: N_EDGE_FEATS=4)
        hidden_channels : width of hidden layers
        n_layers        : number of ResGCNBlocks
        n_classes       : output classes (3: BG/UNK/FG)
        dropout         : dropout rate
        """

        def __init__(
            self,
            in_channels:     int   = N_NODE_FEATS,
            edge_channels:   int   = N_EDGE_FEATS,
            hidden_channels: int   = 128,
            n_layers:        int   = 6,
            n_classes:       int   = 3,
            dropout:         float = 0.2,
        ):
            super().__init__()
            self.n_classes = n_classes

            self.in_norm    = InputNorm(in_channels)
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
            )

            self.blocks = nn.ModuleList([
                ResGCNBlock(hidden_channels, hidden_channels, edge_channels, dropout)
                for _ in range(n_layers)
            ])

            self.head = nn.Sequential(
                nn.Linear(hidden_channels * (n_layers + 1), hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, n_classes),
            )

        def forward(self, data: "Data") -> torch.Tensor:
            x          = data.x
            edge_index = data.edge_index
            edge_attr  = data.edge_attr if data.edge_attr is not None else \
                         torch.zeros(edge_index.size(1), N_EDGE_FEATS, device=x.device)

            h     = self.input_proj(self.in_norm(x))
            all_h = [h]

            for block in self.blocks:
                h = block(h, edge_index, edge_attr)
                all_h.append(h)

            return self.head(torch.cat(all_h, dim=-1))

        @torch.no_grad()
        def predict_probs(self, data: "Data") -> np.ndarray:
            self.eval()
            return F.softmax(self(data), dim=-1).float().cpu().numpy()

        @torch.no_grad()
        def predict_trimap(
            self,
            data: "Data",
            segments: np.ndarray,
            threshold_fg: float = 0.55,
            threshold_bg: float = 0.55,
        ) -> np.ndarray:
            return _probs_to_trimap(self.predict_probs(data), segments,
                                    threshold_fg, threshold_bg)


    # -----------------------------------------------------------------------
    # Model 2: GATTrimapNet
    # -----------------------------------------------------------------------

    class GATTrimapNet(nn.Module):
        """
        Graph Attention Network (GATv2) with edge feature-aware attention.

        GATv2 computes dynamic attention scores that depend on both source
        and destination node features, making it more expressive than GATv1.
        """

        def __init__(
            self,
            in_channels:     int   = N_NODE_FEATS,
            edge_channels:   int   = N_EDGE_FEATS,
            hidden_channels: int   = 128,
            n_heads:         int   = 8,
            n_layers:        int   = 5,
            n_classes:       int   = 3,
            dropout:         float = 0.2,
        ):
            super().__init__()
            self.n_classes  = n_classes
            self.n_heads    = n_heads
            head_dim        = hidden_channels // n_heads

            self.in_norm    = InputNorm(in_channels)
            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )

            self.convs      = nn.ModuleList()
            self.lns        = nn.ModuleList()
            self.edge_gates = nn.ModuleList()

            in_dim = hidden_channels
            for _ in range(n_layers):
                self.convs.append(
                    GATv2Conv(
                        in_dim, head_dim,
                        heads=n_heads, concat=True,
                        dropout=dropout,
                        edge_dim=edge_channels,
                        share_weights=False,
                    )
                )
                out_dim = head_dim * n_heads
                self.lns.append(nn.LayerNorm(out_dim))
                self.edge_gates.append(EdgeInjectionLayer(edge_channels, out_dim))
                in_dim = out_dim

            self.dropout   = dropout
            self.skip_proj = nn.Linear(hidden_channels, in_dim, bias=False)
            self.ctx       = GlobalContextModule(in_dim)

            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, n_classes),
            )

        def forward(self, data: "Data") -> torch.Tensor:
            x          = data.x
            edge_index = data.edge_index
            edge_attr  = data.edge_attr if data.edge_attr is not None else \
                         torch.zeros(edge_index.size(1), N_EDGE_FEATS, device=x.device)

            batch = getattr(data, "batch", None)
            h     = self.input_proj(self.in_norm(x))
            skip  = self.skip_proj(h)

            for conv, ln, eg in zip(self.convs, self.lns, self.edge_gates):
                h_new = conv(h, edge_index, edge_attr)
                h_new = ln(h_new)
                h_new = F.gelu(h_new)
                h_new = F.dropout(h_new, p=self.dropout, training=self.training)
                h_new = eg(edge_attr, edge_index, h_new.size(0), h_new)
                h     = h_new

            h = h + skip
            h = self.ctx(h, batch)
            return self.head(h)

        @torch.no_grad()
        def predict_probs(self, data: "Data") -> np.ndarray:
            self.eval()
            return F.softmax(self(data), dim=-1).float().cpu().numpy()

        @torch.no_grad()
        def predict_trimap(self, data, segments, threshold_fg=0.55, threshold_bg=0.55):
            return _probs_to_trimap(self.predict_probs(data), segments,
                                    threshold_fg, threshold_bg)


    # -----------------------------------------------------------------------
    # Model 3: ResGCNNet  (recommended default)
    # -----------------------------------------------------------------------

    class ResGCNNet(nn.Module):
        """
        Residual GCN with jumping-knowledge fusion — the recommended default.

        Design
        ------
        * Input standardisation, so raw region descriptors need no hand scaling
        * Pre-norm residual blocks (as in Pre-LN Transformers): stable at depth
        * One shared edge encoding, read by every block, in place of a
          per-block edge MLP and scatter
        * Attention-pooled global context, computed per graph so that graphs
          can be batched
        * Jumping-knowledge fusion: a learned convex combination of all block
          outputs, which keeps the head at width D instead of D(n+2) and lets
          the network choose its own effective depth per dataset

        Pipeline
        --------
        InputNorm -> InputProj -> PriorBooster -> [ResBlock x n_layers] ->
        SAGEConv -> JK fusion -> GlobalContext -> Head

        Compared with a dense-concatenation head, fusion by learned weights
        removes the D(n+2) x 2D projection that dominated the parameter count
        and grew with depth; the mixture weights are reported by
        `layer_weights()` and show which propagation depth the trained model
        actually relies on.
        """

        def __init__(
            self,
            in_channels:     int   = N_NODE_FEATS,
            edge_channels:   int   = N_EDGE_FEATS,
            hidden_channels: int   = 128,
            n_layers:        int   = 6,
            n_classes:       int   = 3,
            dropout:         float = 0.15,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.n_layers  = n_layers
            D = hidden_channels

            self.in_norm = InputNorm(in_channels)

            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, D),
                nn.LayerNorm(D),
                nn.GELU(),
            )

            self.prior_booster = nn.Sequential(
                nn.Linear(N_PRIOR_FEATS, max(D // 4, 8)),
                nn.GELU(),
                nn.Linear(max(D // 4, 8), D),
                nn.Sigmoid(),
            )

            self.edge_ctx = EdgeContext(edge_channels, D)

            self.gcn_layers = nn.ModuleList(GCNConv(D, D) for _ in range(n_layers))
            self.norms      = nn.ModuleList(nn.LayerNorm(D) for _ in range(n_layers))

            self.sage      = SAGEConv(D, D)
            self.sage_norm = nn.LayerNorm(D)

            # One mixture weight per fused representation: the projected input,
            # each residual block output, and the coarse SAGE branch.
            self.jk_logits = nn.Parameter(torch.zeros(n_layers + 2))

            self.ctx  = GlobalContextModule(D)
            self.fuse = nn.Sequential(
                nn.LayerNorm(D),
                nn.Linear(D, D),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.head    = nn.Linear(D, n_classes)
            self.dropout = dropout
            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, data: "Data") -> torch.Tensor:
            x          = data.x
            edge_index = data.edge_index
            edge_attr  = data.edge_attr if data.edge_attr is not None else \
                         torch.zeros(edge_index.size(1), N_EDGE_FEATS, device=x.device)
            batch      = getattr(data, "batch", None)
            n_nodes    = x.size(0)

            prior = x[:, -N_PRIOR_FEATS:]      # automatic FG-ness / BG-ness / ambiguity
            h     = self.input_proj(self.in_norm(x))
            h     = h * (1.0 + self.prior_booster(prior))

            gate   = self.edge_ctx(edge_attr, edge_index, n_nodes).to(h.dtype)
            states = [h]

            for gcn, norm in zip(self.gcn_layers, self.norms):
                h_res = gcn(norm(h), edge_index)
                h_res = F.gelu(h_res * gate)
                h_res = F.dropout(h_res, p=self.dropout, training=self.training)
                h     = h + h_res
                states.append(h)

            states.append(F.gelu(self.sage_norm(self.sage(h, edge_index))))

            w    = torch.softmax(self.jk_logits, dim=0).to(h.dtype)
            h_jk = torch.stack(states, dim=0).mul(w[:, None, None]).sum(dim=0)

            h_jk = self.ctx(h_jk, batch)
            return self.head(self.fuse(h_jk))

        @torch.no_grad()
        def layer_weights(self) -> np.ndarray:
            """Fusion weights over [input, block 1..n, SAGE branch]."""
            return torch.softmax(self.jk_logits.detach(), dim=0).cpu().numpy()

        @torch.no_grad()
        def predict_probs(self, data: "Data") -> np.ndarray:
            self.eval()
            return F.softmax(self(data), dim=-1).float().cpu().numpy()

        @torch.no_grad()
        def predict_trimap(
            self,
            data: "Data",
            segments: np.ndarray,
            threshold_fg: float = 0.55,
            threshold_bg: float = 0.55,
        ) -> np.ndarray:
            return _probs_to_trimap(self.predict_probs(data), segments,
                                    threshold_fg, threshold_bg)

        def param_groups(self, base_lr: float) -> list[dict]:
            """
            Layer-wise learning-rate decay: layers closer to the input, whose
            outputs everything downstream depends on, are moved more slowly.
            """
            groups = []
            n = self.n_layers
            for i, (gcn, norm) in enumerate(zip(self.gcn_layers, self.norms)):
                groups.append({
                    "params": list(gcn.parameters()) + list(norm.parameters()),
                    "lr": base_lr * (0.8 ** (n - i)),
                })
            groups.append({
                "params": (list(self.in_norm.parameters()) +
                           list(self.input_proj.parameters()) +
                           list(self.prior_booster.parameters())),
                "lr": base_lr * 0.5,
            })
            groups.append({
                "params": (list(self.edge_ctx.parameters()) +
                           list(self.sage.parameters()) +
                           list(self.sage_norm.parameters()) +
                           list(self.ctx.parameters())),
                "lr": base_lr * 0.9,
            })
            groups.append({
                "params": ([self.jk_logits] +
                           list(self.fuse.parameters()) +
                           list(self.head.parameters())),
                "lr": base_lr,
            })
            return groups


    def build_model(
        variant:         str   = "resgcn",
        in_channels:     int   = N_NODE_FEATS,
        edge_channels:   int   = N_EDGE_FEATS,
        hidden_channels: int   = 128,
        n_layers:        int   = 6,
        n_classes:       int   = 3,
        dropout:         float = 0.2,
    ) -> "nn.Module":
        """
        Factory to select model variant by name.

        variant : "resgcn" | "gcn" | "gat"
        """
        kw = dict(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            n_classes=n_classes,
            dropout=dropout,
        )
        if variant == "resgcn":
            return ResGCNNet(**kw, n_layers=n_layers)
        if variant == "gat":
            return GATTrimapNet(**kw, n_layers=n_layers, n_heads=8)
        if variant == "gcn":
            return GCNTrimapNet(**kw, n_layers=n_layers)
        raise ValueError(f"Unknown variant '{variant}'. Choose: resgcn | gcn | gat")


def probs_to_node_trimap(
    probs:        np.ndarray,
    threshold_fg: float = 0.55,
    threshold_bg: float = 0.55,
) -> np.ndarray:
    """
    Map per-region class probabilities to the four GrabCut labels.

    A region is declared definite only when the corresponding probability
    clears its threshold; otherwise the more likely of the two sides is
    handed to GrabCut as a probable label, leaving it free to move the
    boundary within that region.

    Returns
    -------
    node_labels : (N,) uint8 in {GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD}
    """
    bg_p, fg_p = probs[:, CLASS_BG], probs[:, CLASS_FG]

    labels = np.where(fg_p > bg_p, TRIMAP_PROB_FG, TRIMAP_PROB_BG).astype(np.uint8)
    labels[bg_p >= threshold_bg] = TRIMAP_BG
    labels[fg_p >= threshold_fg] = TRIMAP_FG
    return labels


def project_to_pixels(node_values: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Broadcast a per-region quantity to pixels through the label map.

    Indexing the region axis with the label map replaces one boolean scan of
    the image per region, turning an O(N·H·W) projection into O(H·W).
    """
    n_needed = int(segments.max()) + 1
    values   = node_values
    if values.shape[0] < n_needed:
        pad = np.zeros((n_needed - values.shape[0], *values.shape[1:]),
                       dtype=values.dtype)
        values = np.concatenate([values, pad], axis=0)
    return values[segments]


def _probs_to_trimap(
    probs:        np.ndarray,
    segments:     np.ndarray,
    threshold_fg: float,
    threshold_bg: float,
) -> np.ndarray:
    """Convert per-superpixel class probabilities to a pixel-level trimap."""
    node_labels = probs_to_node_trimap(probs, threshold_fg, threshold_bg)
    if node_labels.shape[0] < int(segments.max()) + 1:
        node_labels = np.concatenate([
            node_labels,
            np.full(int(segments.max()) + 1 - node_labels.shape[0],
                    TRIMAP_PROB_BG, dtype=np.uint8),
        ])
    return node_labels[segments].astype(np.uint8)