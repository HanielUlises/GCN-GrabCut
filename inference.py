"""
inference.py — Automatic segmentation with GCN-GrabCut.

No clicks, no windows, no user interaction: give it an image (or a folder of
images) and it writes out the masks.

Usage
-----
    python3 inference.py --image path/to/image.jpg
    python3 inference.py --input path/to/folder --output results/
    python3 inference.py --image cat.jpg --keep-largest --save mask overlay
"""

import argparse
import time
import cv2
import torch
from pathlib import Path

from src.gcn_grabcut import GCNGrabCutPipeline
from src.gcn_grabcut.graph_builder import SuperpixelGraphConfig
from src.gcn_grabcut.model import ResGCNNet, GCNTrimapNet, GATTrimapNet

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

parser = argparse.ArgumentParser(description="Automatic GCN-GrabCut segmentation")
src = parser.add_mutually_exclusive_group(required=True)
src.add_argument("--image", help="Path to a single input image")
src.add_argument("--input", help="Directory of images to segment")

parser.add_argument("--output",     default="results", help="Output directory")
parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
parser.add_argument("--model",      default="resgcn", choices=["resgcn", "gcn", "gat"])
parser.add_argument("--device",     default="cuda")
parser.add_argument("--threshold",  type=float, default=0.55,
                    help="Softmax threshold for definite FG/BG superpixels")
parser.add_argument("--max-size",   type=int, default=800,
                    help="Resize longest edge to this before segmenting (0 = off)")
parser.add_argument("--refine",     type=int, default=0,
                    help="Extra GrabCut refinement iterations")
parser.add_argument("--superpixels", type=int, default=300)
parser.add_argument("--hidden",     type=int, default=128)
parser.add_argument("--layers",     type=int, default=6)
parser.add_argument("--no-edge-aware", action="store_true",
                    help="Threshold region probabilities directly instead of "
                         "projecting them through a guided filter")
parser.add_argument("--filter-radius", type=int, default=8,
                    help="Guided-filter radius for the edge-aware trimap")
parser.add_argument("--min-area",   type=float, default=0.002,
                    help="Drop mask components smaller than this fraction of the image")
parser.add_argument("--keep-largest", action="store_true",
                    help="Keep only the largest connected component")
parser.add_argument("--save", nargs="+", default=["mask", "overlay"],
                    choices=["mask", "overlay", "rgba", "trimap"],
                    help="Which outputs to write")
args = parser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    args.device = "cpu"
if args.device == "mps" and not torch.backends.mps.is_available():
    args.device = "cpu"

# --------------------------------------------------------------------- model

ckpt_path = Path(args.checkpoint)
if not ckpt_path.exists():
    fallback = Path("checkpoints/final_model.pt")
    if not fallback.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path} (or {fallback}). Train one with train.py."
        )
    print(f"[inference] {ckpt_path} not found, using {fallback}")
    ckpt_path = fallback

model_cls = {"resgcn": ResGCNNet, "gcn": GCNTrimapNet, "gat": GATTrimapNet}[args.model]
ckpt  = torch.load(ckpt_path, map_location=args.device, weights_only=True)
state = ckpt["model"]

# Width and depth are recovered from the checkpoint so that a model trained
# with non-default settings loads without repeating them on the command line.
hidden = (state["input_proj.0.weight"].shape[0]
          if "input_proj.0.weight" in state else args.hidden)
layers = (sum(1 for k in state if k.startswith("gcn_layers.") and k.endswith(".bias"))
          or sum(1 for k in state if k.startswith("blocks.") and k.endswith(".conv.bias"))
          or sum(1 for k in state if k.startswith("convs.") and k.endswith(".bias"))
          or args.layers)

model = model_cls(hidden_channels=hidden, n_layers=layers)
model.load_state_dict(state)
model.eval()
print(f"[inference] loaded {model_cls.__name__} (D={hidden}, n={layers}) "
      f"from {ckpt_path} on {args.device}")

pipeline = GCNGrabCutPipeline(
    model,
    sp_config=SuperpixelGraphConfig(n_segments=args.superpixels),
    device=args.device,
)

# --------------------------------------------------------------------- inputs

if args.image:
    paths = [Path(args.image)]
else:
    in_dir = Path(args.input)
    paths  = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not paths:
        raise FileNotFoundError(f"No images found in {in_dir}")

out_dir = Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------- run

total_t = 0.0
n_done  = 0

for path in paths:
    image = cv2.imread(str(path))
    if image is None:
        print(f"[inference] skipping unreadable file: {path}")
        continue

    if args.max_size > 0:
        h, w  = image.shape[:2]
        scale = min(args.max_size / max(h, w), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)

    t0 = time.perf_counter()
    result = pipeline.segment(
        image,
        threshold_fg=args.threshold,
        threshold_bg=args.threshold,
        refine_iters=args.refine,
        min_area_ratio=args.min_area,
        keep_largest=args.keep_largest,
        edge_aware=not args.no_edge_aware,
        filter_radius=args.filter_radius,
    )
    elapsed = time.perf_counter() - t0
    total_t += elapsed
    n_done  += 1

    stem = out_dir / path.stem
    if "mask" in args.save:
        cv2.imwrite(f"{stem}_mask.png", result.binary_mask * 255)
    if "overlay" in args.save:
        cv2.imwrite(f"{stem}_overlay.png", result.overlay)
    if "rgba" in args.save:
        cv2.imwrite(f"{stem}_rgba.png", result.rgba)
    if "trimap" in args.save:
        from src.gcn_grabcut.pipeline import _colour_trimap
        cv2.imwrite(f"{stem}_trimap.png", _colour_trimap(result.trimap))

    t = result.timing
    print(f"[{n_done}/{len(paths)}] {path.name}  "
          f"fg={result.binary_mask.mean():.1%}  "
          f"graph={t.get('graph_build', 0):.2f}s "
          f"gcn={t.get('gcn_inference', 0):.3f}s "
          f"grabcut={t.get('grabcut', 0):.2f}s  total={elapsed:.2f}s")

if n_done:
    print(f"\n[inference] {n_done} image(s) → {out_dir}/  "
          f"({total_t / n_done:.2f}s per image)")
else:
    print("[inference] nothing to do.")
