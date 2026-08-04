"""
Generate the GCN-GrabCut architecture figure.

The figure is monochrome and typeset in a serif face so that it reads like a
diagram from a journal article rather than a slide. Panel (a)-(e) trace one
image through the pipeline; the lower row expands the network itself.

    python3 tools/make_architecture_figure.py            # writes gcn_architecture.png
    python3 tools/make_architecture_figure.py --pdf      # also writes a vector copy
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

LW       = 0.9
FS_STAGE = 10.0
FS_BODY  = 8.6
FS_MATH  = 8.4
FS_SMALL = 7.2

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "mathtext.fontset":  "dejavuserif",
    "text.color":        INK,
    "savefig.facecolor": "white",
})


def box(ax, x, y, w, h, fill=FILL_LIGHT, radius=0.55, lw=LW, zorder=2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=INK, facecolor=fill, zorder=zorder,
    ))
    return x + w / 2, y + h / 2


def arrow(ax, p, q, ls="solid", color=INK, lw=LW, rad=0.0, mut=7.0,
          style="-|>", zorder=4):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=mut,
        linewidth=lw, edgecolor=color, facecolor=color,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=2.0, shrinkB=2.0, zorder=zorder,
    ))


def draw_image_panel(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor=FILL_LIGHT, zorder=2))
    th = np.linspace(0, 2 * np.pi, 240)
    cx, cy, rx, ry = x + 0.46 * w, y + 0.44 * h, 0.26 * w, 0.30 * h
    ax.fill(cx + rx * np.cos(th), cy + ry * np.sin(th),
            facecolor=FILL_DARK, edgecolor=INK, linewidth=LW * 0.8, zorder=3)
    ax.plot([x, x + w], [y + 0.79 * h] * 2, color=INK_SOFT,
            linewidth=LW * 0.7, zorder=3)


def draw_graph_panel(ax, x, y, w, h, seed=11):
    """Superpixel tessellation with the region-adjacency graph on top."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor="white", zorder=2))

    rng = np.random.default_rng(seed)
    nx_, ny_ = 4, 4
    pts, inside = [], []
    cx, cy, rx, ry = x + 0.46 * w, y + 0.44 * h, 0.27 * w, 0.31 * h
    for i in range(nx_):
        for j in range(ny_):
            px = x + w * (i + 0.5 + rng.uniform(-0.18, 0.18)) / nx_
            py = y + h * (j + 0.5 + rng.uniform(-0.18, 0.18)) / ny_
            pts.append((px, py))
            inside.append(((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 < 1.0)
    pts = np.asarray(pts)

    for i in range(1, nx_):
        ax.plot([x + i * w / nx_] * 2, [y, y + h], color=FILL_DARK,
                linewidth=LW * 0.55, zorder=3)
    for j in range(1, ny_):
        ax.plot([x, x + w], [y + j * h / ny_] * 2, color=FILL_DARK,
                linewidth=LW * 0.55, zorder=3)

    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    drawn = set()
    for i in range(len(pts)):
        for j in np.argsort(d[i])[:2]:
            key = (min(i, j), max(i, j))
            if key not in drawn:
                drawn.add(key)
                ax.plot(*zip(pts[i], pts[j]), color=INK_SOFT,
                        linewidth=LW * 0.7, zorder=4)

    # Two non-local edges: colour-similar regions that are not adjacent.
    for i, j in ((0, 13), (3, 11)):
        ax.plot(*zip(pts[i], pts[j]), color=INK_SOFT, linewidth=LW * 0.7,
                linestyle=(0, (2.2, 1.8)), zorder=4)

    for (px, py), ins in zip(pts, inside):
        ax.add_patch(Circle((px, py), 0.42, linewidth=LW * 0.8, edgecolor=INK,
                            facecolor=FILL_DARK if ins else "white", zorder=5))


def draw_prob_panel(ax, x, y, w, h):
    """Region-level foreground probability, before any thresholding."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor="white", zorder=2))
    rng = np.random.default_rng(3)
    n = 7
    cx, cy = 0.46, 0.44
    for i in range(n):
        for j in range(n):
            u, v = (i + 0.5) / n, (j + 0.5) / n
            r = np.hypot((u - cx) / 0.30, (v - cy) / 0.34)
            p = float(np.clip(1.15 - r + rng.normal(0, 0.07), 0, 1))
            shade = str(round(0.96 - 0.86 * p, 3))
            ax.add_patch(Rectangle((x + i * w / n, y + j * h / n),
                                   w / n, h / n, linewidth=0.0,
                                   facecolor=shade, zorder=3))
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor="none", zorder=5))


def draw_trimap_panel(ax, x, y, w, h):
    """Definite foreground, unknown band, definite background."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor=FILL_LIGHT, zorder=2))
    th = np.linspace(0, 2 * np.pi, 240)
    cx, cy, rx, ry = x + 0.46 * w, y + 0.44 * h, 0.29 * w, 0.33 * h
    wob = 1 + 0.05 * np.sin(5 * th)
    ax.fill(cx + 1.30 * rx * wob * np.cos(th), cy + 1.28 * ry * wob * np.sin(th),
            facecolor=FILL_MID, edgecolor=INK, linewidth=LW * 0.7,
            linestyle=(0, (2, 1.6)), zorder=3)
    ax.fill(cx + rx * wob * np.cos(th), cy + ry * wob * np.sin(th),
            facecolor="#5a5a5a", edgecolor=INK, linewidth=LW * 0.7, zorder=4)


def draw_mask_panel(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=LW,
                           edgecolor=INK, facecolor="white", zorder=2))
    th = np.linspace(0, 2 * np.pi, 240)
    cx, cy, rx, ry = x + 0.46 * w, y + 0.44 * h, 0.28 * w, 0.32 * h
    r = 1 + 0.045 * np.sin(6 * th)
    ax.fill(cx + rx * r * np.cos(th), cy + ry * r * np.sin(th),
            facecolor=INK, edgecolor=INK, linewidth=LW, zorder=3)


def build(outputs: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    top_y, top_h, pw = 41.0, 12.0, 11.6
    xs = [2.0, 21.4, 40.8, 60.2, 79.6]

    titles = ["Input image", "Superpixel graph", "Region posteriors",
              "Trimap", "Segmentation"]
    formulas = [
        r"$I \in \mathbb{R}^{H\times W\times 3}$",
        r"$\mathcal{G}=(\mathcal{V},\mathcal{E})$,"
        r"$\;\mathbf{x}_i\!\in\!\mathbb{R}^{19}$,$\;\mathbf{e}_{ij}\!\in\!\mathbb{R}^{5}$",
        r"$P(c\,|\,v_i)$, $\;c \in \{$BG, UNK, FG$\}$",
        r"$T \in \{0,1,2,3\}^{H\times W}$",
        r"$M \in \{0,1\}^{H\times W}$",
    ]
    painters = [draw_image_panel, draw_graph_panel, draw_prob_panel,
                draw_trimap_panel, draw_mask_panel]

    for k, (x, title, formula, paint) in enumerate(zip(xs, titles, formulas, painters)):
        ax.text(x, top_y + top_h + 1.6, f"({'abcde'[k]})  {title}",
                ha="left", va="bottom", fontsize=FS_STAGE)
        paint(ax, x, top_y, pw, top_h)
        ax.text(x + pw / 2, top_y - 1.8, formula, ha="center", va="top",
                fontsize=FS_MATH)

    mid = top_y + top_h / 2
    ops = ["SLIC and\nauto prior", "message\npassing", "guided\nfilter",
           "GrabCut and\nclean-up"]
    for k, label in enumerate(ops):
        a, b = xs[k] + pw, xs[k + 1]
        arrow(ax, (a + 0.4, mid), (b - 0.4, mid))
        ax.text((a + b) / 2, mid + 0.7, label, ha="center", va="bottom",
                fontsize=FS_SMALL, color=INK_SOFT, linespacing=1.4)

    ax.plot([2, 98], [36.0, 36.0], color=FILL_DARK, linewidth=LW * 0.8)
    ax.text(2, 33.0,
            r"Detail of the trimap network $f_\theta$ "
            r"(hidden width $D$, $n$ residual blocks)",
            ha="left", va="center", fontsize=FS_STAGE)

    by, bh = 14.0, 8.2
    stages = [
        (2.0,  13.0, "Input norm\nand projection",
         r"$\mathbb{R}^{19} \rightarrow \mathbb{R}^{D}$"),
        (17.5, 12.0, "Prior booster",
         r"$\mathbf{h}\odot(1+\sigma(\mathrm{MLP}(\mathbf{p}_i)))$"),
        (32.0, 24.0, "Residual GCN block   $\\times\\,n$", None),
        (58.4, 13.4, "Multi-scale branch",
         r"SAGEConv$(D \rightarrow D)$"),
        (74.2, 11.2, "Depth fusion",
         r"$\sum_k w_k\,\mathbf{h}^{(k)}$"),
        (87.6, 12.4, "Context and head",
         r"$\mathbb{R}^{D} \rightarrow \mathbb{R}^{3}$"),
    ]

    centres = []
    for x, w, title, sub in stages:
        fill = FILL_MID if "Residual" in title else FILL_LIGHT
        box(ax, x, by, w, bh, fill=fill)
        ax.text(x + w / 2, by + bh - 1.9, title, ha="center", va="center",
                fontsize=FS_BODY, linespacing=1.3)
        if sub is not None:
            ax.text(x + w / 2, by + 1.9, sub, ha="center", va="center",
                    fontsize=FS_SMALL, color=INK_SOFT)
        centres.append((x, w))

    rx0, rw = 32.0, 24.0
    inner = [("LayerNorm", 6.0), ("GCNConv", 5.4), (r"$\odot\;\mathbf{g}_i$", 3.4),
             ("GELU", 3.6)]
    ix = rx0 + 0.8
    for text, iw in inner:
        box(ax, ix, by + 1.0, iw, 3.0, fill="white", radius=0.35, lw=LW * 0.8)
        ax.text(ix + iw / 2, by + 2.5, text, ha="center", va="center",
                fontsize=6.6)
        ix += iw + 0.35
    ax.text(rx0 + rw - 1.1, by + 2.5, r"$+$", ha="center", va="center", fontsize=10)
    arrow(ax, (rx0 + 0.8, by + 4.3), (rx0 + rw - 1.1, by + 3.8),
          ls=(0, (2.5, 2)), color=INK_SOFT, lw=LW * 0.8, rad=-0.10, mut=5.5)

    for k in range(len(centres) - 1):
        x, w = centres[k]
        arrow(ax, (x + w + 0.4, by + bh / 2), (centres[k + 1][0] - 0.4, by + bh / 2))

    ex, ew, eh = 32.0, 24.0, 4.4
    ey = by - 7.6
    box(ax, ex, ey, ew, eh, fill="white", radius=0.45, lw=LW * 0.8)
    ax.text(ex + ew / 2, ey + eh / 2 + 0.85, "Edge context, computed once",
            ha="center", va="center", fontsize=FS_SMALL)
    ax.text(ex + ew / 2, ey + eh / 2 - 1.0,
            r"$\mathbf{g}_i=\sigma\!\left(W\,\mathrm{mean}_{j\in\mathcal{N}(i)}"
            r"\,\phi(\mathbf{e}_{ij})\right)$",
            ha="center", va="center", fontsize=FS_SMALL, color=INK_SOFT)
    arrow(ax, (ex + ew / 2, ey + eh), (ex + ew * 0.52, by),
          color=INK_SOFT, lw=LW * 0.8, mut=6.0)

    # The fusion stage reads every intermediate representation; the taps are
    # drawn as a bus above the row so they do not cross the stage boxes.
    fx, fw = centres[4]
    bus = by + bh + 2.6
    tap_x = [x + w / 2 for x, w in centres[:4]]
    ax.plot([tap_x[0], fx + fw / 2], [bus, bus], color=INK_SOFT,
            linewidth=LW * 0.75, linestyle=(0, (2.2, 2.0)), zorder=3)
    for x in tap_x:
        ax.plot([x, x], [by + bh, bus], color=INK_SOFT, linewidth=LW * 0.75,
                linestyle=(0, (2.2, 2.0)), zorder=3)
    arrow(ax, (fx + fw / 2, bus), (fx + fw / 2, by + bh),
          color=INK_SOFT, lw=LW * 0.75, mut=6.0)
    ax.text((tap_x[0] + fx) / 2, bus + 1.4,
            "learned convex combination of the projected input, "
            "every block output and the coarse branch",
            ha="center", va="bottom", fontsize=FS_SMALL, color=INK_SOFT,
            style="italic")

    ax.text(50, 3.0,
            "Node features: 16 region descriptors (LAB and HSV statistics, "
            "position, area, isoperimetric ratio, gradient, boundary ratio) "
            "$\\Vert$ 3 automatic prior channels.\n"
            "Edge features: colour dissimilarity, centroid distance, shared "
            "boundary length, gradient contrast, edge type; in (b) solid edges "
            "are region adjacencies and dashed edges non-local colour links.\n"
            "Trimap labels follow OpenCV: $0$ definite background, $1$ definite "
            "foreground, $2$ probable background, $3$ probable foreground.",
            ha="center", va="center", fontsize=FS_SMALL, color=INK_SOFT,
            linespacing=1.55)

    for out in outputs:
        fig.savefig(out, bbox_inches="tight", pad_inches=0.14, facecolor="white")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gcn_architecture.png")
    ap.add_argument("--pdf", action="store_true", help="also emit a vector copy")
    a = ap.parse_args()

    targets = [Path(a.out)]
    if a.pdf:
        targets.append(Path(a.out).with_suffix(".pdf"))
    build(targets)
