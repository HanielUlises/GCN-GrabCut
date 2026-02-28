"""
inference.py — Run GCN-GrabCut on any image
Usage:
    python3 inference.py --image path/to/image.jpg

Then click the image:
    Left click  = foreground (what you want to keep)
    Right click = background (what you want to remove)
    Press SPACE or ENTER when done clicking → runs segmentation
    Press R to reset clicks
    Press Q to quit
"""

import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

from src.gcn_grabcut import GCNGrabCutPipeline
from src.gcn_grabcut.model import ResGCNNet

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--image",      required=True,          help="Path to input image")
parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", help="Model checkpoint")
parser.add_argument("--device",     default="cuda")
parser.add_argument("--output",     default="output",       help="Output filename prefix")
parser.add_argument("--threshold",  type=float, default=0.55)
args = parser.parse_args()

ckpt_path = Path(args.checkpoint)
if not ckpt_path.exists():
    ckpt_path = Path("checkpoints/final_model.pt")
    print(f"[info] best_model.pt not found, using {ckpt_path}")

if args.device == "cuda" and not torch.cuda.is_available():
    args.device = "cpu"

print(f"[inference] loading model from {ckpt_path}...")
model = ResGCNNet()
ckpt  = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(ckpt["model"])
model.eval()
print("[inference] model loaded!")

pipeline = GCNGrabCutPipeline(model, device=args.device)


image = cv2.imread(args.image)
if image is None:
    raise FileNotFoundError(f"Could not read image: {args.image}")

MAX_DIM = 800
h, w = image.shape[:2]
scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
display = cv2.resize(image, (int(w * scale), int(h * scale)))


fg_points = []
bg_points = []

COLORS = {"fg": (0, 255, 0), "bg": (0, 0, 255)}

def draw_canvas():
    canvas = display.copy()
    for (r, c) in fg_points:
        cv2.circle(canvas, (c, r), 6, COLORS["fg"], -1)
    for (r, c) in bg_points:
        cv2.circle(canvas, (c, r), 6, COLORS["bg"], -1)
    cv2.putText(canvas, "Left=FG  Right=BG  SPACE=run  R=reset  Q=quit",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.imshow("GCN-GrabCut", canvas)

def mouse_cb(event, x, y, flags, param):
    r = int(y / scale)
    c = int(x / scale)
    if event == cv2.EVENT_LBUTTONDOWN:
        fg_points.append((r, c))
        draw_canvas()
    elif event == cv2.EVENT_RBUTTONDOWN:
        bg_points.append((r, c))
        draw_canvas()

cv2.namedWindow("GCN-GrabCut")
cv2.setMouseCallback("GCN-GrabCut", mouse_cb)
draw_canvas()

print("\n[inference] Click the image:")
print("  Left click  = foreground (green)")
print("  Right click = background (red)")
print("  SPACE/ENTER = run segmentation")
print("  R           = reset clicks")
print("  Q           = quit\n")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('q'):
        print("[inference] quit.")
        break

    elif key == ord('r'):
        fg_points.clear()
        bg_points.clear()
        draw_canvas()
        print("[inference] clicks reset.")

    elif key in (32, 13):
        if not fg_points and not bg_points:
            print("[inference] add at least one click first!")
            continue

        print(f"[inference] running with {len(fg_points)} FG + {len(bg_points)} BG clicks...")
        result = pipeline.segment(
            image,
            fg_points=fg_points,
            bg_points=bg_points,
            threshold_fg=args.threshold,
            threshold_bg=args.threshold,
        )

        t = result.timing
        print(f"[inference] done! graph={t.get('graph_build',0):.2f}s  "
              f"gcn={t.get('gcn_inference',0):.3f}s  "
              f"grabcut={t.get('grabcut',0):.3f}s")

        result.save(args.output)
        print(f"[inference] saved → {args.output}_overlay.png / _rgba.png / _mask.png")

        result.show()
        draw_canvas()

cv2.destroyAllWindows()