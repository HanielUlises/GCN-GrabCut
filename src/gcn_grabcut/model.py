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
* Edge features (4-dim) are projected and fused into node updates
* Global context: a graph-level summary vector is broadcast back to each node
* Layer-wise learning-rate decay via parameter groups
* Label propagation from user hint nodes via hint-conditioned message passing
* 3-class output: {BG=0, UNKNOWN=1, FG=2}

Node input: 19-dim  (16 image features + 3 hint features)
Edge input: 4-dim

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
    from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, global_mean_pool
    from torch_geometric.data import Data
    _TORCH_GEOMETRIC = True
except ImportError:
    _TORCH_GEOMETRIC = False

from .graph_builder import N_NODE_FEATS, N_EDGE_FEATS


TRIMAP_BG      = 0   # cv2.GC_BGD
TRIMAP_FG      = 1   # cv2.GC_FGD
TRIMAP_PROB_BG = 2   # cv2.GC_PR_BGD
TRIMAP_PROB_FG = 3   # cv2.GC_PR_FGD

CLASS_BG  = 0
CLASS_UNK = 1
CLASS_FG  = 2


if _TORCH:

    class EdgeInjectionLayer(nn.Module):
        """
        Projects edge features and uses them to gate/bias node updates.
        This allows the model to weight messages differently based on
        colour dissimilarity, spatial distance, etc.
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
            """
            Gate the aggregated node messages using projected edge attributes.

            Parameters
            ----------
            edge_attr    : (E, edge_dim)
            edge_index   : (2, E)
            n_nodes      : int
            node_updates : (N, hidden_dim) — raw aggregated messages

            Returns
            -------
            gated : (N, hidden_dim)
            """
            gates = self.proj(edge_attr)
            dst   = edge_index[1]
            gate_agg = torch.zeros(n_nodes, gates.size(1), device=gates.device)
            gate_agg.scatter_add_(0, dst.unsqueeze(1).expand_as(gates), gates)

            count = torch.bincount(dst, minlength=n_nodes).float().clamp(min=1)
            gate_agg = gate_agg / count.unsqueeze(1)
            return node_updates * gate_agg


    class GlobalContextModule(nn.Module):
        """
        Computes a graph-level summary and broadcasts it back to all nodes.
        This gives every node access to global image context.
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.compress = nn.Linear(hidden_dim, hidden_dim // 2)
            self.expand   = nn.Linear(hidden_dim // 2, hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x : (N, hidden_dim) → returns (N, hidden_dim) context-enriched"""
            g = x.mean(dim=0, keepdim=True)
            g = F.relu(self.compress(g))
            g = torch.sigmoid(self.expand(g))
            return x * g


    class ResGCNBlock(nn.Module):
        """
        One residual GCN block:
          x' = BN(ReLU(GCNConv(x))) + skip(x)
          x' = EdgeInjection(x')
          x' = GlobalContext(x')
        """
        def __init__(self, in_dim: int, out_dim: int, edge_dim: int, dropout: float):
            super().__init__()
            self.conv        = GCNConv(in_dim, out_dim)
            self.bn          = nn.BatchNorm1d(out_dim)
            self.dropout     = dropout
            self.skip        = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()
            self.edge_inject = EdgeInjectionLayer(edge_dim, out_dim)
            self.ctx         = GlobalContextModule(out_dim)

        def forward(self, x, edge_index, edge_attr):
            h = self.conv(x, edge_index)
            h = self.bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + self.skip(x)
            h = self.edge_inject(edge_attr, edge_index, x.size(0), h)
            h = self.ctx(h)
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

            h = self.input_proj(x)
            all_h = [h]

            for block in self.blocks:
                h = block(h, edge_index, edge_attr)
                all_h.append(h)

            h_cat = torch.cat(all_h, dim=-1)
            return self.head(h_cat)

        @torch.no_grad()
        def predict_trimap(
            self,
            data: "Data",
            segments: np.ndarray,
            threshold_fg: float = 0.55,
            threshold_bg: float = 0.55,
        ) -> np.ndarray:
            """Infer and upsample to pixel trimap."""
            self.eval()
            logits = self(data)
            probs  = F.softmax(logits, dim=-1).cpu().numpy()
            return _probs_to_trimap(probs, segments, threshold_fg, threshold_bg)


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

            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
            )

            self.convs = nn.ModuleList()
            self.lns   = nn.ModuleList()
            self.edge_gates = nn.ModuleList()

            in_dim = hidden_channels
            for i in range(n_layers):
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

            self.skip_proj = nn.Linear(hidden_channels, in_dim, bias=False)
            self.ctx  = GlobalContextModule(in_dim)

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

            h    = self.input_proj(x)
            skip = self.skip_proj(h)

            for conv, ln, eg in zip(self.convs, self.lns, self.edge_gates):
                h_new = conv(h, edge_index, edge_attr)
                h_new = ln(h_new)
                h_new = F.gelu(h_new)
                h_new = F.dropout(h_new, p=0.2, training=self.training)
                h_new = eg(edge_attr, edge_index, x.size(0), h_new)
                h = h_new

            h = h + skip
            h = self.ctx(h)
            return self.head(h)

        @torch.no_grad()
        def predict_trimap(self, data, segments, threshold_fg=0.55, threshold_bg=0.55):
            self.eval()
            probs = F.softmax(self(data), dim=-1).cpu().numpy()
            return _probs_to_trimap(probs, segments, threshold_fg, threshold_bg)


    # -----------------------------------------------------------------------
    # Model 3: ResGCNNet  (recommended default)
    # -----------------------------------------------------------------------

    class ResGCNNet(nn.Module):
        """
        Deep Residual GCN — best default for GCN-GrabCut research.

        Design
        ------
        * Pre-norm residual blocks (like Pre-LN Transformers): more stable training
        * Multi-scale aggregation: runs GCN at two scales (fine + coarse features)
        * Hint-conditioned attention: FG/BG hint nodes get boosted attention
        * Dense connections: every block output is concatenated for final prediction

        Architecture
        ------------
        InputProj → [ResBlock × n_layers] → DenseConcat → MultiScaleFusion → Head

        This is the most expressive model and is recommended when you have ≥100 images.
        For small datasets (< 50 images), use GCNTrimapNet.
        """

        def __init__(
            self,
            in_channels:     int   = N_NODE_FEATS,
            edge_channels:   int   = N_EDGE_FEATS,
            hidden_channels: int   = 128,
            n_layers:        int   = 8,
            n_classes:       int   = 3,
            dropout:         float = 0.15,
        ):
            super().__init__()
            self.n_classes = n_classes
            self.n_layers  = n_layers
            D = hidden_channels

            self.input_proj = nn.Sequential(
                nn.Linear(in_channels, D),
                nn.LayerNorm(D),
                nn.GELU(),
            )

            self.hint_booster = nn.Sequential(
                nn.Linear(3, D // 4),
                nn.ReLU(),
                nn.Linear(D // 4, D),
                nn.Sigmoid(),
            )

            self.gcn_layers = nn.ModuleList()
            self.norms      = nn.ModuleList()
            self.edge_projs = nn.ModuleList()
            for _ in range(n_layers):
                self.gcn_layers.append(GCNConv(D, D))
                self.norms.append(nn.LayerNorm(D))
                self.edge_projs.append(nn.Sequential(
                    nn.Linear(edge_channels, D),
                    nn.ReLU(),
                    nn.Linear(D, D),
                    nn.Sigmoid(),
                ))

            self.sage = SAGEConv(D, D)
            self.sage_norm = nn.LayerNorm(D)

            self.global_proj = nn.Linear(D, D)
            self.global_gate = nn.Linear(D, D)

            dense_in = D * (n_layers + 2)
            self.fusion = nn.Sequential(
                nn.Linear(dense_in, D * 2),
                nn.LayerNorm(D * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(D * 2, D),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            )

            self.head = nn.Linear(D, n_classes)

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

            hints = x[:, -3:]
            h     = self.input_proj(x)
            hint_gate = self.hint_booster(hints)
            h = h * (1.0 + hint_gate)

            dense_outputs = [h]

            for gcn, norm, ep in zip(self.gcn_layers, self.norms, self.edge_projs):
                h_res = norm(h)
                h_res = gcn(h_res, edge_index)

                edge_gate = ep(edge_attr)
                dst       = edge_index[1]
                N         = h.size(0)
                gate_sum  = torch.zeros(N, h.size(1), device=h.device)
                gate_sum.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_gate), edge_gate)
                counts    = torch.bincount(dst, minlength=N).float().clamp(1)
                gate_avg  = gate_sum / counts.unsqueeze(1)
                h_res     = h_res * gate_avg

                h_res = F.gelu(h_res)
                h_res = F.dropout(h_res, p=self.dropout, training=self.training)
                h     = h + h_res
                dense_outputs.append(h)

            h_sage = F.gelu(self.sage_norm(self.sage(h, edge_index)))
            dense_outputs.append(h_sage)

            # Global context injection
            g_mean = h.mean(dim=0, keepdim=True)
            g_ctx  = torch.sigmoid(self.global_gate(
                F.gelu(self.global_proj(g_mean))
            ))
            dense_outputs = [d * g_ctx for d in dense_outputs]

            # Dense concatenation + head
            h_dense = torch.cat(dense_outputs, dim=-1)      # (N, D*(n+2))
            h_fused = self.fusion(h_dense)
            return self.head(h_fused)

        @torch.no_grad()
        def predict_trimap(
            self,
            data: "Data",
            segments: np.ndarray,
            threshold_fg: float = 0.55,
            threshold_bg: float = 0.55,
        ) -> np.ndarray:
            self.eval()
            probs = F.softmax(self(data), dim=-1).cpu().numpy()
            return _probs_to_trimap(probs, segments, threshold_fg, threshold_bg)

        def param_groups(self, base_lr: float) -> list[dict]:
            """
            Layer-wise learning rate decay.
            Earlier layers get lower LR (common practice for GNNs).
            """
            groups = []
            n = self.n_layers
            for i, (gcn, norm, ep) in enumerate(
                zip(self.gcn_layers, self.norms, self.edge_projs)
            ):
                decay = 0.8 ** (n - i)
                groups.append({"params": list(gcn.parameters()) +
                                         list(norm.parameters()) +
                                         list(ep.parameters()),
                               "lr": base_lr * decay})
            groups.append({"params": list(self.input_proj.parameters()) +
                                     list(self.hint_booster.parameters()),
                           "lr": base_lr * 0.5})
            groups.append({"params": list(self.fusion.parameters()) +
                                     list(self.head.parameters()),
                           "lr": base_lr})
            groups.append({"params": list(self.sage.parameters()) +
                                     list(self.sage_norm.parameters()) +
                                     list(self.global_proj.parameters()) +
                                     list(self.global_gate.parameters()),
                           "lr": base_lr * 0.9})
            return groups


    def build_model(
        variant:         str   = "resgcn",
        in_channels:     int   = N_NODE_FEATS,
        edge_channels:   int   = N_EDGE_FEATS,
        hidden_channels: int   = 128,
        n_layers:        int   = 6,
        n_classes:       int   = 3,
        dropout:         float = 0.2,
    ) -> nn.Module:
        """
        Factory to select model variant by name.

        Parameters

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




def _probs_to_trimap(
    probs:        np.ndarray,    # (N, 3) [P(BG), P(UNK), P(FG)]
    segments:     np.ndarray,    # (H, W)
    threshold_fg: float,
    threshold_bg: float,
) -> np.ndarray:
    """Convert per-superpixel class probabilities to pixel-level trimap."""
    H, W   = segments.shape
    trimap = np.full((H, W), TRIMAP_PROB_BG, dtype=np.uint8)

    for nid in range(probs.shape[0]):
        mask = segments == nid
        if not mask.any():
            continue
        bg_p, unk_p, fg_p = probs[nid]
        if fg_p >= threshold_fg:
            trimap[mask] = TRIMAP_FG
        elif bg_p >= threshold_bg:
            trimap[mask] = TRIMAP_BG
        elif fg_p > bg_p:
            trimap[mask] = TRIMAP_PROB_FG
        else:
            trimap[mask] = TRIMAP_PROB_BG

    return trimap
