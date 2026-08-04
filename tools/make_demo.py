"""
Render a demonstration of the pipeline running on real images.

For each input image the five stages of the pipeline are shown in sequence —
image, superpixel graph, foreground posterior, trimap, and the refined mask —
with the wall-clock cost of the run. Output is an animated GIF for inline
display and an H.264 video.

    python3 tools/make_demo.py --checkpoint checkpoints_duts/best_model.pt \
        --input /path/to/images --n 5 --out demo
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import mark_boundaries

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn_grabcut.graph_builder import GraphBuilder, SuperpixelGraphConfig
from src.gcn_grabcut.grabcut import Label
from src.gcn_grabcut.model import build_model, project_to_pixels, CLASS_FG
from src.gcn_grabcut.metrics import evaluate
from src.gcn_grabcut.pipeline import GCNGrabCutPipeline

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
INK      = (28, 28, 28)
INK_SOFT = (112, 112, 112)
PAPER    = (255, 255, 255)
RULE     = (208, 208, 208)

PANEL_W, PANEL_H = 760, 500
MARGIN, HEAD_H, FOOT_H = 28, 62, 52
CANVAS = (PANEL_W + 2 * MARGIN, PANEL_H + HEAD_H + FOOT_H + 2 * MARGIN)


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / name
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_TITLE = _font("DejaVuSerif.ttf", 24)
F_SUB   = _font("DejaVuSerif-Italic.ttf", 15)
F_FOOT  = _font("DejaVuSerif.ttf", 14)


def fit(img: np.ndarray) -> np.ndarray:
    """Scale a BGR image into the panel box, letterboxed on white."""
    h, w = img.shape[:2]
    s = min(PANEL_W / w, PANEL_H / h)
    out = np.full((PANEL_H, PANEL_W, 3), 255, np.uint8)
    r = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)),
                   interpolation=cv2.INTER_AREA)
    y0 = (PANEL_H - r.shape[0]) // 2
    x0 = (PANEL_W - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def compose(panel: np.ndarray, index: int, total: int,
            title: str, subtitle: str, footer: str) -> Image.Image:
    """Place one stage panel on the page furniture."""
    page = Image.new("RGB", CANVAS, PAPER)
    d = ImageDraw.Draw(page)

    d.text((MARGIN, MARGIN - 4), f"({'abcde'[index]})  {title}",
           font=F_TITLE, fill=INK)
    d.text((MARGIN, MARGIN + 30), subtitle, font=F_SUB, fill=INK_SOFT)

    top = MARGIN + HEAD_H
    page.paste(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)),
               (MARGIN, top))
    d.rectangle([MARGIN, top, MARGIN + PANEL_W - 1, top + PANEL_H - 1],
                outline=RULE)

    base = top + PANEL_H + 18
    d.line([MARGIN, base - 8, MARGIN + PANEL_W, base - 8], fill=RULE)
    d.text((MARGIN, base), footer, font=F_FOOT, fill=INK_SOFT)

    # Stage indicator, kept in the header where no caption can run into it.
    r, gap = 5, 18
    cx = MARGIN + PANEL_W - (total - 1) * gap - r
    cy = MARGIN + 12
    for k in range(total):
        bounds = [cx + k * gap - r, cy - r, cx + k * gap + r, cy + r]
        d.ellipse(bounds, fill=INK if k <= index else PAPER, outline=INK_SOFT)
    return page


PAIR_W, PAIR_H = 372, 430


def cutout(image: np.ndarray, mask: np.ndarray, contour=True) -> np.ndarray:
    """Dim everything outside the mask and outline what is kept."""
    out = image.copy()
    keep = mask.astype(bool)
    out[~keep] = (255 * 0.90 + out[~keep] * 0.10).astype(np.uint8)
    if contour:
        cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (30, 30, 30), 2)
    return out


def fit_box(img: np.ndarray, w: int, h: int) -> np.ndarray:
    hh, ww = img.shape[:2]
    s = min(w / ww, h / hh)
    out = np.full((h, w, 3), 255, np.uint8)
    r = cv2.resize(img, (max(int(ww * s), 1), max(int(hh * s), 1)),
                   interpolation=cv2.INTER_AREA)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return out


def compose_pair(left: np.ndarray, right: np.ndarray, title: str,
                 subtitle: str, footer: str) -> Image.Image:
    """Two panels side by side: what the method produced, and the truth."""
    page = Image.new("RGB", CANVAS, PAPER)
    d = ImageDraw.Draw(page)
    d.text((MARGIN, MARGIN - 4), title, font=F_TITLE, fill=INK)
    d.text((MARGIN, MARGIN + 30), subtitle, font=F_SUB, fill=INK_SOFT)

    # Centred inside the same panel area the stage frames use, so every frame
    # of the recording has identical geometry.
    top = MARGIN + HEAD_H + (PANEL_H - PAIR_H) // 2 - 10
    gap = PANEL_W - 2 * PAIR_W
    for k, (panel, label) in enumerate(((left, "predicted"), (right, "ground truth"))):
        x = MARGIN + k * (PAIR_W + gap)
        page.paste(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), (x, top))
        d.rectangle([x, top, x + PAIR_W - 1, top + PAIR_H - 1], outline=RULE)
        d.text((x, top + PAIR_H + 6), label, font=F_FOOT, fill=INK_SOFT)

    base = top + PANEL_H + 18
    d.line([MARGIN, base - 8, MARGIN + PANEL_W, base - 8], fill=RULE)
    d.text((MARGIN, base), footer, font=F_FOOT, fill=INK_SOFT)
    return page


def stage_panels(image: np.ndarray, model, pipeline, sp_cfg, segment_kw=None):
    """Run the pipeline once and return the five stage visualisations."""
    from torch_geometric.data import Data as PyGData

    graph = GraphBuilder(image, sp_cfg).build()
    data = PyGData(
        x=torch.tensor(graph.node_input(), dtype=torch.float32),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_attr, dtype=torch.float32),
    )
    probs = model.predict_probs(data)
    result = pipeline.segment(image, **(segment_kw or {}))

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    sp_vis = (mark_boundaries(rgb, graph.segments, color=(0.1, 0.1, 0.1),
                              mode="thick") * 255).astype(np.uint8)
    sp_vis = cv2.cvtColor(sp_vis, cv2.COLOR_RGB2BGR)

    p_fg = project_to_pixels(probs[:, CLASS_FG].astype(np.float32), graph.segments)
    post = cv2.cvtColor((255 * (1.0 - np.clip(p_fg, 0, 1))).astype(np.uint8),
                        cv2.COLOR_GRAY2BGR)

    tri = np.zeros(result.trimap.shape, np.uint8)
    tri[result.trimap == Label.BG_DEFINITE] = 245
    tri[result.trimap == Label.BG_PROBABLE] = 200
    tri[result.trimap == Label.FG_PROBABLE] = 110
    tri[result.trimap == Label.FG_DEFINITE] = 40
    tri = cv2.cvtColor(tri, cv2.COLOR_GRAY2BGR)

    mask = result.binary_mask.astype(bool)
    cut = image.copy()
    cut[~mask] = (255 * 0.93 + cut[~mask] * 0.07).astype(np.uint8)
    contours, _ = cv2.findContours(result.binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(cut, contours, -1, (30, 30, 30), 2)

    t = result.timing
    total = sum(t.values())
    captions = [
        ("Input image",
         "no bounding box, scribble, or trimap is supplied",
         f"{image.shape[1]}x{image.shape[0]} pixels"),
        ("Superpixel graph",
         f"{graph.n_nodes} SLIC regions, {graph.n_edges // 2} undirected edges",
         f"graph construction {t.get('graph_build', 0):.2f} s"),
        ("Foreground posterior",
         "per-region P(FG | v) predicted by the residual GCN; dark is foreground",
         f"network {t.get('gcn_inference', 0) * 1000:.0f} ms"),
        ("Trimap",
         "posteriors projected to pixels through a guided filter, then thresholded",
         "four labels: definite and probable foreground and background"),
        ("Segmentation",
         "GrabCut refinement of the predicted trimap, followed by clean-up",
         f"GrabCut {t.get('grabcut', 0):.2f} s   ·   total {total:.2f} s"),
    ]
    return [sp for sp in (image, sp_vis, post, tri, cut)], captions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints_duts/best_model.pt")
    ap.add_argument("--model", default="resgcn", choices=["resgcn", "gcn", "gat"])
    ap.add_argument("--input", required=True, help="Directory of demo images")
    ap.add_argument("--n", type=int, default=5,
                    help="Images shown stage by stage")
    ap.add_argument("--masks", default=None,
                    help="Mask directory; enables the ground-truth gallery")
    ap.add_argument("--gallery", default=None,
                    help="Directory of images for the results gallery")
    ap.add_argument("--gallery-n", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--filter-radius", type=int, default=4)
    ap.add_argument("--keep-largest", action="store_true")
    ap.add_argument("--out", default="demo", help="Output basename")
    ap.add_argument("--superpixels", type=int, default=500)
    ap.add_argument("--max-size", type=int, default=512)
    ap.add_argument("--hold", type=float, default=1.3,
                    help="Seconds each stage is held")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu",
                       weights_only=True)["model"]
    hidden = state["input_proj.0.weight"].shape[0]
    layers = (sum(1 for k in state if k.startswith("gcn_layers.") and k.endswith(".bias"))
              or sum(1 for k in state if k.startswith("blocks.") and k.endswith(".conv.bias"))
              or sum(1 for k in state if k.startswith("convs.") and k.endswith(".bias")))
    model = build_model(args.model, hidden_channels=hidden, n_layers=layers)
    model.load_state_dict(state)
    model.eval()

    sp_cfg = SuperpixelGraphConfig(n_segments=args.superpixels)
    pipeline = GCNGrabCutPipeline(model, sp_cfg, device="cpu")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted(p for p in Path(args.input).iterdir()
                   if p.suffix.lower() in exts)[:args.n]
    if not paths:
        raise SystemExit(f"no images found in {args.input}")

    segment_kw = dict(threshold_fg=args.threshold, threshold_bg=args.threshold,
                      filter_radius=args.filter_radius,
                      keep_largest=args.keep_largest)

    tmp = Path(tempfile.mkdtemp(prefix="gcn_demo_"))
    n_frames = max(1, int(round(args.hold * args.fps)))
    frame_id = 0

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        h, w = image.shape[:2]
        s = min(args.max_size / max(h, w), 1.0)
        if s < 1.0:
            image = cv2.resize(image, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)

        panels, captions = stage_panels(image, model, pipeline, sp_cfg, segment_kw)
        for k, (panel, (title, sub, foot)) in enumerate(zip(panels, captions)):
            page = compose(fit(panel), k, len(panels), title, sub,
                           f"{path.name}   ·   {foot}")
            hold = n_frames * (2 if k == len(panels) - 1 else 1)
            for _ in range(hold):
                page.save(tmp / f"f{frame_id:05d}.png")
                frame_id += 1
        print(f"[demo] {path.name}")

    if args.gallery and args.masks:
        gal_dir = Path(args.gallery)
        gal_paths = sorted(p for p in gal_dir.iterdir()
                           if p.suffix.lower() in exts)[:args.gallery_n]
        hold = max(1, int(round(0.95 * args.fps)))
        for k, path in enumerate(gal_paths):
            image = cv2.imread(str(path))
            gt_path = next((Path(args.masks) / (path.stem + e)
                            for e in (".png", ".jpg", ".bmp")
                            if (Path(args.masks) / (path.stem + e)).exists()), None)
            if image is None or gt_path is None:
                continue
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            h, w = image.shape[:2]
            s = min(args.max_size / max(h, w), 1.0)
            if s < 1.0:
                image = cv2.resize(image, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
                gt = cv2.resize(gt, (image.shape[1], image.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            gt = (gt > 127).astype(np.uint8)

            result = pipeline.segment(image, **segment_kw)
            iou = evaluate(result.binary_mask, gt).iou
            page = compose_pair(
                fit_box(cutout(image, result.binary_mask), PAIR_W, PAIR_H),
                fit_box(cutout(image, gt), PAIR_W, PAIR_H),
                f"Result {k + 1} of {len(gal_paths)}",
                "the method sees only the image; the mask on the right is the "
                "annotation it is scored against",
                f"{path.name}   ·   IoU {iou:.2f}   ·   "
                f"{sum(result.timing.values()):.2f} s",
            )
            for _ in range(hold):
                page.save(tmp / f"f{frame_id:05d}.png")
                frame_id += 1
            print(f"[demo] gallery {path.name} IoU={iou:.3f}")

    pattern = str(tmp / "f%05d.png")
    mp4 = f"{args.out}.mp4"
    gif = f"{args.out}.gif"

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4],
        check=True)

    palette = tmp / "palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", pattern, "-vf", "fps=5,scale=720:-1:flags=lanczos,palettegen",
         str(palette)], check=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
         "-i", pattern, "-i", str(palette),
         "-lavfi", "fps=5,scale=720:-1:flags=lanczos[v];[v][1:v]paletteuse",
         gif], check=True)

    shutil.rmtree(tmp, ignore_errors=True)
    for f in (mp4, gif):
        print(f"wrote {f}  ({Path(f).stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
