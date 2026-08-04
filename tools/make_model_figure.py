"""
Generate the ResGCNNet diagram: the trimap network on its own.

The left column is the computation, the right column shows what each stage does
to the graph itself — the same nine-node example is carried from the attributed
input through one aggregation step and the graph readout to the classified
output.

    python3 tools/make_model_figure.py             # writes gcn_model.png
    python3 tools/make_model_figure.py --pdf       # also writes a vector copy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

INK        = "#1b1b1b"
INK_SOFT   = "#707070"
FILL_LIGHT = "#f5f5f5"
FILL_MID   = "#e6e6e6"
FILL_DARK  = "#c6c6c6"
NODE_FG    = "#4a4a4a"
NODE_UNK   = "#b0b0b0"

LW       = 0.9
FS_TITLE = 10.0
FS_BODY  = 8.6
FS_MATH  = 8.0
FS_SMALL = 7.2
FS_TINY  = 6.4

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "mathtext.fontset":  "dejavuserif",
    "text.color":        INK,
    "savefig.facecolor": "white",
})

# One fixed example graph, reused by every vignette so that the reader can follow
# the same nine regions through the network. Coordinates are in vignette units.
NODES = np.array([
    [0.16, 0.80], [0.46, 0.90], [0.78, 0.78],
    [0.10, 0.50], [0.44, 0.56], [0.80, 0.46],
    [0.22, 0.18], [0.54, 0.20], [0.86, 0.14],
])
ADJ = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5),
       (3, 6), (4, 7), (5, 8), (6, 7), (7, 8)]
NONLOCAL = [(0, 8), (2, 6)]
CENTRE = 4                          # the node whose update is illustrated


def box(ax, x, y, w, h, fill=FILL_LIGHT, radius=0.5, lw=LW, ls="solid", zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=INK, facecolor=fill, linestyle=ls, zorder=zorder,
    ))
    return x + w / 2, y + h / 2


def arrow(ax, p, q, ls="solid", color=INK, lw=LW, rad=0.0, mut=7.0,
          style="-|>", zorder=5):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, edgecolor=color, facecolor=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2.0, shrinkB=2.0, zorder=zorder,
    ))


class Vignette:
    """Maps the unit square of the example graph into a region of the figure."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def at(self, i):
        u, v = NODES[i]
        return self.x + u * self.w, self.y + v * self.h

    def frame(self, ax, label=None):
        ax.add_patch(Rectangle((self.x - 1.4, self.y - 1.4), self.w + 2.8,
                               self.h + 2.8, linewidth=LW * 0.7,
                               edgecolor=FILL_DARK, facecolor="white", zorder=1))
        if label:
            ax.text(self.x - 1.0, self.y + self.h + 2.5, label, ha="left",
                    va="center", fontsize=FS_SMALL, color=INK_SOFT, style="italic")

    def edges(self, ax, nonlocal_too=True, color=INK_SOFT, lw=None):
        lw = lw or LW * 0.7
        for i, j in ADJ:
            ax.plot(*zip(self.at(i), self.at(j)), color=color, linewidth=lw,
                    zorder=3)
        if nonlocal_too:
            for i, j in NONLOCAL:
                ax.plot(*zip(self.at(i), self.at(j)), color=color, linewidth=lw,
                        linestyle=(0, (2.2, 1.8)), zorder=3)

    def nodes(self, ax, r=1.15, fills=None, edge=INK, lw=None, zorder=4):
        lw = lw or LW * 0.8
        for i in range(len(NODES)):
            f = "white" if fills is None else fills[i]
            ax.add_patch(Circle(self.at(i), r, linewidth=lw, edgecolor=edge,
                                facecolor=f, zorder=zorder))


def attribute_stack(ax, x, y, n, w=1.15, h=1.05, filled=0, label=None):
    """A short column of cells standing for an attribute vector."""
    for k in range(n):
        ax.add_patch(Rectangle((x, y + k * h), w, h, linewidth=LW * 0.6,
                               edgecolor=INK, zorder=6,
                               facecolor=FILL_MID if k < filled else "white"))
    if label:
        ax.text(x + w + 0.9, y + n * h / 2, label, ha="left", va="center",
                fontsize=FS_TINY, color=INK_SOFT)


def stage(ax, x, y, w, h, label, formula, shape, fill=FILL_LIGHT):
    box(ax, x, y, w, h, fill=fill)
    ax.text(x + w / 2, y + h / 2 + (1.05 if formula else 0.0), label,
            ha="center", va="center", fontsize=FS_BODY)
    if formula:
        ax.text(x + w / 2, y + h / 2 - 1.55, formula, ha="center", va="center",
                fontsize=FS_MATH, color=INK_SOFT)
    if shape:
        ax.text(x + w + 1.4, y + h / 2, shape, ha="left", va="center",
                fontsize=FS_SMALL, color=INK_SOFT)


def build(outputs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 13.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 130)
    ax.axis("off")

    ax.text(4, 126.6,
            r"$\mathbf{ResGCNNet}$: region posteriors from an attributed graph",
            ha="left", va="center", fontsize=FS_TITLE + 1)
    ax.text(4, 123.4,
            r"hidden width $D$, depth $n$; the nine-node graph on the right is the "
            r"same example at every stage",
            ha="left", va="center", fontsize=FS_SMALL, color=INK_SOFT,
            style="italic")

    x0, w0 = 4.0, 32.0
    cx = x0 + w0 / 2

    rows = [
        (110.0, 6.6, "Node attributes",
         r"$\mathbf{x}_i = [\varphi(S_i) \Vert \pi_i]$", r"$N \times 19$", FILL_MID),
        (101.5, 6.0, "Input standardisation",
         r"$(\mathbf{x}_i - m)/\sqrt{v+\epsilon}$", r"$N \times 19$", FILL_LIGHT),
        (93.0, 6.0, "Linear, LayerNorm, GELU",
         r"$W_{\mathrm{in}} \hat{\mathbf{x}}_i$", r"$N \times D$", FILL_LIGHT),
        (84.5, 6.0, "Prior gate",
         r"$\odot\,(1 + \sigma(\mathrm{MLP}_\pi(\pi_i)))$", r"$N \times D$", FILL_LIGHT),
        (68.0, 11.5, r"Residual GCN block, $\ell = 1 \ldots n$", None,
         r"$N \times D$", FILL_MID),
        (58.5, 6.0, "Coarse branch (SAGEConv)",
         r"$W_1 \mathbf{h}^{(n)}_i + W_2 \overline{\mathbf{h}^{(n)}_{\mathcal{N}(i)}}$",
         r"$N \times D$", FILL_LIGHT),
        (48.5, 7.0, "Depth fusion",
         r"$\mathbf{z}_i = \sum_{k=0}^{n+1} w_k \mathbf{h}^{(k)}_i$",
         r"$N \times D$", FILL_MID),
        (38.5, 6.0, "Global context gate",
         r"$\mathbf{z}_i \odot \sigma(W_e\,\mathrm{ReLU}(W_c \mathbf{s}))$",
         None, FILL_LIGHT),
        (30.0, 6.0, "Head", "LayerNorm, Linear, GELU, Linear",
         r"$N \times 3$", FILL_LIGHT),
        (21.0, 6.2, "Softmax over classes",
         r"$P(c \mid v_i)$", r"$N \times 3$", FILL_MID),
    ]

    for y, h, label, formula, shape, fill in rows:
        stage(ax, x0, y, w0, h, label, formula, shape, fill=fill)
    for k in range(len(rows) - 1):
        arrow(ax, (cx, rows[k][0]), (cx, rows[k + 1][0] + rows[k + 1][1]))

    # The attributed graph.
    v1 = Vignette(50.0, 104.0, 28.0, 15.0)
    v1.frame(ax, "the attributed graph")
    v1.edges(ax)
    v1.nodes(ax)
    marked = 5          # a mid-height node, so its attribute stack fits inside
    p = v1.at(marked)
    ax.add_patch(Circle(p, 1.15, linewidth=LW * 1.5, edgecolor=INK,
                        facecolor=FILL_MID, zorder=5))
    attribute_stack(ax, p[0] + 2.2, p[1] - 2.6, 5, filled=2,
                    label=r"$\mathbf{x}_i \in \mathbb{R}^{19}$")
    em = ((v1.at(3)[0] + v1.at(4)[0]) / 2, (v1.at(3)[1] + v1.at(4)[1]) / 2)
    ax.text(em[0], em[1] - 2.4, r"$\mathbf{e}_{ij} \in \mathbb{R}^{5}$",
            ha="center", va="center", fontsize=FS_TINY, color=INK_SOFT)
    ax.text(v1.x - 1.0, v1.y - 3.4,
            "solid: region adjacency        dashed: non-local colour edge",
            ha="left", va="center", fontsize=FS_TINY, color=INK_SOFT)
    arrow(ax, (x0 + w0 + 5.8, 110.0 + 3.3), (v1.x - 2.6, v1.y + v1.h * 0.5),
          style="-", ls=(0, (2, 2)), color=FILL_DARK, lw=LW * 0.8)

    # Edge context, read by every block.
    ex, ew, eh = 44.0, 44.0, 8.4
    ey = 88.6
    box(ax, ex, ey, ew, eh, fill=FILL_LIGHT)
    ax.text(ex + ew / 2, ey + eh - 2.0, "Edge context, computed once",
            ha="center", va="center", fontsize=FS_BODY)
    ax.text(ex + ew / 2, ey + 2.4,
            r"$\mathbf{g}_i = \sigma(W_g\,\mathrm{LN}(|\mathcal{N}(i)|^{-1}"
            r"\sum_{j \in \mathcal{N}(i)} \phi_e(\mathbf{e}_{ij})))$",
            ha="center", va="center", fontsize=FS_MATH, color=INK_SOFT)
    ax.text(ex + ew + 1.4, ey + eh / 2, r"$N \times D$", ha="left", va="center",
            fontsize=FS_SMALL, color=INK_SOFT)
    ax.text(ex + ew * 0.30 - 1.6, ey - 2.4, "edge attributes enter here only",
            ha="right", va="center", fontsize=FS_TINY, color=INK_SOFT,
            style="italic")

    # One aggregation step, drawn on the example graph.
    bx, bw = 44.0, 44.0
    by, bh = 61.0, 19.0
    box(ax, bx, by, bw, bh, fill="white", lw=LW * 0.8, ls=(0, (3, 2)))
    ax.text(bx + bw - 1.2, by + bh - 1.7, "one residual block",
            ha="right", va="center", fontsize=FS_SMALL, color=INK_SOFT,
            style="italic")
    arrow(ax, (ex + ew * 0.30, ey), (bx + bw * 0.30, by + bh),
          color=INK_SOFT, lw=LW * 0.9, mut=6.0)
    arrow(ax, (x0 + w0 + 5.8, by + bh * 0.5), (bx - 0.6, by + bh * 0.5),
          style="-", ls=(0, (2, 2)), color=FILL_DARK, lw=LW * 0.8)

    v2 = Vignette(bx + 2.2, by + 5.6, 15.5, 9.6)
    v2.edges(ax, color=FILL_DARK)
    v2.nodes(ax, r=1.0)
    c = v2.at(CENTRE)
    for i, j in ADJ:
        if CENTRE in (i, j):
            other = j if i == CENTRE else i
            arrow(ax, v2.at(other), c, color=INK, lw=LW * 0.75, mut=5.0, zorder=6)
    ax.add_patch(Circle(c, 1.3, linewidth=LW * 1.6, edgecolor=INK,
                        facecolor=FILL_MID, zorder=7))
    ax.text(c[0] + 2.0, c[1] - 0.2, r"$v_i$", ha="left", va="center",
            fontsize=FS_TINY, zorder=8)
    ax.text(v2.x - 0.4, v2.y + v2.h + 2.6,
            r"neighbours summed with weights $(\hat{d}_i \hat{d}_j)^{-1/2}$,",
            ha="left", va="center", fontsize=FS_TINY, color=INK_SOFT)
    ax.text(v2.x - 0.4, v2.y + v2.h + 0.8,
            r"then scaled channel-wise by $\mathbf{g}_i$ and added back",
            ha="left", va="center", fontsize=FS_TINY, color=INK_SOFT)

    ax.text(bx + bw * 0.66, by + bh - 6.0,
            r"$\mathbf{u}^{(\ell)}_i = \sum_{j \in \mathcal{N}(i) \cup \{i\}}"
            r"(\hat{d}_i \hat{d}_j)^{-1/2} W^{(\ell)} \mathrm{LN}(\mathbf{h}^{(\ell-1)}_j)$",
            ha="center", va="center", fontsize=FS_MATH, color=INK_SOFT)
    ax.text(bx + bw * 0.66, by + bh - 10.4,
            r"$\mathbf{h}^{(\ell)}_i = \mathbf{h}^{(\ell-1)}_i +"
            r"\mathrm{Drop}(\mathrm{GELU}(\mathbf{g}_i \odot \mathbf{u}^{(\ell)}_i))$",
            ha="center", va="center", fontsize=FS_MATH)

    inner = [("LayerNorm", 8.0), ("GCNConv", 7.4), (r"$\odot\,\mathbf{g}_i$", 4.6),
             ("GELU", 5.4)]
    ix, iy, ih = bx + 2.2, by + 1.3, 3.2
    for text, iw in inner:
        box(ax, ix, iy, iw, ih, fill=FILL_LIGHT, radius=0.3, lw=LW * 0.75)
        ax.text(ix + iw / 2, iy + ih / 2, text, ha="center", va="center",
                fontsize=FS_TINY)
        if ix + iw + 0.9 < bx + bw - 12.0:
            arrow(ax, (ix + iw, iy + ih / 2), (ix + iw + 0.9, iy + ih / 2),
                  mut=4.5, lw=LW * 0.7)
        ix += iw + 0.9
    ax.text(ix + 0.9, iy + ih / 2, r"$+\;\mathbf{h}^{(\ell-1)}_i$", ha="left",
            va="center", fontsize=FS_TINY)

    # Fusion bus: every representation reaches the fusion stage.
    bus_x = x0 - 2.4
    fuse_y = 48.5 + 7.0 / 2
    ax.plot([bus_x, bus_x], [fuse_y, 96.0], color=INK_SOFT, linewidth=LW * 0.8,
            linestyle=(0, (2.4, 2.0)), zorder=3)
    for y in (96.0, 68.0 + 5.8, 58.5 + 3.0):
        ax.plot([bus_x, x0], [y, y], color=INK_SOFT, linewidth=LW * 0.8,
                linestyle=(0, (2.4, 2.0)), zorder=3)
    arrow(ax, (bus_x, fuse_y), (x0, fuse_y), color=INK_SOFT, lw=LW * 0.8, mut=6.0)
    ax.text(bus_x - 1.3, (fuse_y + 96.0) / 2, "all $n+2$ representations",
            rotation=90, ha="center", va="center", fontsize=FS_SMALL,
            color=INK_SOFT, style="italic")

    # Graph readout: attention over nodes into one vector.
    sx, sw, sh = 44.0, 44.0, 15.4
    sy = 34.6
    box(ax, sx, sy, sw, sh, fill="white", lw=LW * 0.8, ls=(0, (3, 2)))
    ax.text(sx + sw - 1.2, sy + sh - 1.7, "graph readout, per graph",
            ha="right", va="center", fontsize=FS_SMALL, color=INK_SOFT,
            style="italic")
    v3 = Vignette(sx + 2.2, sy + 3.4, 14.5, 8.8)
    v3.edges(ax, color=FILL_DARK)
    rng = np.random.default_rng(4)
    weights = np.clip(rng.uniform(0.15, 1.0, len(NODES)), 0, 1)
    weights[CENTRE] = 1.0
    v3.nodes(ax, r=1.0, fills=[str(round(0.97 - 0.72 * w, 3)) for w in weights])
    pool = (sx + 26.0, sy + sh / 2 - 0.5)
    for i in range(len(NODES)):
        arrow(ax, v3.at(i), pool, color=INK_SOFT, lw=LW * 0.55, mut=3.6,
              rad=0.08, zorder=4)
    attribute_stack(ax, pool[0], pool[1] - 2.6, 5, filled=3,
                    label=r"$\mathbf{s} \in \mathbb{R}^{D}$")
    ax.text(sx + 2.0, sy + 1.4,
            r"$\alpha_i = \mathrm{softmax}_i(\mathbf{u}^\top \mathbf{z}_i)$,"
            r"$\;\; \mathbf{s} = \sum_i \alpha_i \mathbf{z}_i$;"
            r"  node shade is $\alpha_i$",
            ha="left", va="center", fontsize=FS_TINY, color=INK_SOFT)
    arrow(ax, (sx, sy + sh * 0.45), (x0 + w0 + 0.6, 38.5 + 3.0),
          color=INK_SOFT, lw=LW * 0.9, mut=6.0)
    ax.text(x0 + w0 + 1.4, 38.5 + 6.0 + 1.4, r"$N \times D$", ha="left",
            va="center", fontsize=FS_SMALL, color=INK_SOFT)

    # The classified graph.
    v4 = Vignette(50.0, 19.5, 21.0, 10.5)
    v4.frame(ax, "the classified graph")
    v4.edges(ax, nonlocal_too=False, color=FILL_DARK)
    cls = [0, 0, 0, 1, 2, 1, 2, 2, 0]
    palette = ["white", NODE_UNK, NODE_FG]
    v4.nodes(ax, r=1.15, fills=[palette[c] for c in cls])
    lx, ly = v4.x + v4.w + 3.0, v4.y + v4.h - 1.2
    for k, (fill, name) in enumerate((("white", "background"),
                                      (NODE_UNK, "unknown"),
                                      (NODE_FG, "foreground"))):
        ax.add_patch(Circle((lx, ly - k * 3.6), 1.0, linewidth=LW * 0.8,
                            edgecolor=INK, facecolor=fill, zorder=5))
        ax.text(lx + 1.9, ly - k * 3.6, name, ha="left", va="center",
                fontsize=FS_TINY, color=INK_SOFT)
    arrow(ax, (x0 + w0 + 5.8, 21.0 + 3.1), (v4.x - 2.6, v4.y + v4.h * 0.5),
          style="-", ls=(0, (2, 2)), color=FILL_DARK, lw=LW * 0.8)

    ax.text(4, 11.0,
            "Solid arrows carry node representations; dashed arrows carry a "
            "quantity read by more than one stage.\nEvery graph-level reduction "
            "is taken per graph, so several graphs occupy one batch without "
            "interacting, and the\nper-node output is the same whether a graph "
            "is evaluated alone or in company.",
            ha="left", va="center", fontsize=FS_SMALL, color=INK_SOFT,
            linespacing=1.6)

    for out in outputs:
        fig.savefig(out, bbox_inches="tight", pad_inches=0.14, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gcn_model.png")
    ap.add_argument("--pdf", action="store_true", help="also emit a vector copy")
    a = ap.parse_args()

    targets = [Path(a.out)]
    if a.pdf:
        targets.append(Path(a.out).with_suffix(".pdf"))
    build(targets)
