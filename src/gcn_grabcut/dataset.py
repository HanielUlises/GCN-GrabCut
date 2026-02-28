"""
Dataset utilities for GCN-GrabCut training.

Responsibilities
----------------
1. Load image/mask pairs from disk.
2. Simulate user interaction (random click sampling).
3. Apply augmentation (flip, colour jitter, random rotation, crop).
4. Pre-compute superpixel graphs + derive per-superpixel trimap labels.
5. Cache processed graphs to speed up subsequent epochs.

Sample dict schema
------------------
{
  "image"     : np.ndarray  (H, W, 3) BGR uint8
  "gt_mask"   : np.ndarray  (H, W)    uint8 binary {0, 1}
  "fg_points" : list[(row, col)]
  "bg_points" : list[(row, col)]
  "name"      : str
}

Processed tuple: (PyG Data, labels tensor (N,), segments array (H,W))
"""

from __future__ import annotations

import numpy as np
import cv2
import random
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.data import Data
    _TORCH = True
except ImportError:
    _TORCH = False

from .graph_builder import GraphBuilder, SuperpixelGraphConfig, encode_user_hints
from .model import CLASS_BG, CLASS_UNK, CLASS_FG, N_NODE_FEATS


# -----------------------------------------------------------------------
# Click simulation
# -----------------------------------------------------------------------

def sample_clicks(
    gt_mask:        np.ndarray,
    n_fg:           int   = 5,
    n_bg:           int   = 5,
    erosion_radius: int   = 8,
    jitter:         float = 0.0,
) -> tuple[list, list]:
    """
    Randomly sample foreground and background click coordinates.

    Parameters
    ----------
    gt_mask        : binary uint8 (H, W)
    n_fg / n_bg    : desired number of clicks per class
    erosion_radius : erode mask before sampling to avoid boundary clicks
    jitter         : fraction of image diagonal to randomly perturb each click

    Returns
    -------
    fg_points, bg_points : list of (row, col)
    """
    kernel    = np.ones((erosion_radius * 2 + 1,) * 2, np.uint8)
    fg_region = cv2.erode(gt_mask, kernel)
    bg_region = cv2.erode(1 - gt_mask, kernel)

    H, W = gt_mask.shape
    diag = np.sqrt(H**2 + W**2)

    def _sample(region, n):
        coords = np.argwhere(region > 0)
        if len(coords) == 0:
            return []
        idx = np.random.choice(len(coords), min(n, len(coords)), replace=False)
        pts = coords[idx].tolist()
        if jitter > 0:
            pts_jit = []
            for r, c in pts:
                dr = int(np.random.randn() * jitter * diag)
                dc = int(np.random.randn() * jitter * diag)
                r2 = int(np.clip(r + dr, 0, H - 1))
                c2 = int(np.clip(c + dc, 0, W - 1))
                pts_jit.append((r2, c2))
            return pts_jit
        return [(int(r), int(c)) for r, c in pts]

    return _sample(fg_region, n_fg), _sample(bg_region, n_bg)


# -----------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------

def augment_sample(
    image: np.ndarray,
    mask:  np.ndarray,
    prob_flip:   float = 0.5,
    prob_rotate: float = 0.3,
    prob_color:  float = 0.5,
    prob_crop:   float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply stochastic augmentation to an image/mask pair.

    Returns
    -------
    aug_image, aug_mask  (same dtype as input)
    """
    H, W = image.shape[:2]

    # 1. Horizontal flip
    if random.random() < prob_flip:
        image = cv2.flip(image, 1)
        mask  = cv2.flip(mask,  1)

    # 2. Random rotation [-15, 15] degrees
    if random.random() < prob_rotate:
        angle = random.uniform(-15, 15)
        M     = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (W, H),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask  = cv2.warpAffine(mask.astype(np.uint8), M, (W, H),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    # 3. Colour jitter (brightness, contrast, saturation)
    if random.random() < prob_color:
        image = _color_jitter(image)

    # 4. Random crop and resize (zoom in/out)
    if random.random() < prob_crop:
        scale = random.uniform(0.75, 1.0)
        ch    = int(H * scale)
        cw    = int(W * scale)
        y0    = random.randint(0, H - ch)
        x0    = random.randint(0, W - cw)
        image = cv2.resize(image[y0:y0+ch, x0:x0+cw], (W, H), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask [y0:y0+ch, x0:x0+cw], (W, H), interpolation=cv2.INTER_NEAREST)

    return image, mask


def _color_jitter(image: np.ndarray) -> np.ndarray:
    """Random brightness + contrast + saturation jitter."""
    img = image.astype(np.float32)
    # Brightness
    delta = random.uniform(-40, 40)
    img   = np.clip(img + delta, 0, 255)
    # Contrast
    factor = random.uniform(0.7, 1.3)
    img    = np.clip(128 + factor * (img - 128), 0, 255)
    # Saturation (in HSV)
    hsv  = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    img  = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    return img.astype(np.uint8)


# -----------------------------------------------------------------------
# Per-superpixel label derivation
# -----------------------------------------------------------------------

def derive_trimap_labels(
    segments:      np.ndarray,
    gt_mask:       np.ndarray,
    fg_threshold:  float = 0.75,
    bg_threshold:  float = 0.75,
) -> np.ndarray:
    """
    Assign a 3-class trimap label to each superpixel by majority vote.

    Label assignment
    ----------------
    fg_ratio ≥ fg_threshold → CLASS_FG  (2)
    fg_ratio ≤ 1-bg_threshold → CLASS_BG  (0)
    otherwise → CLASS_UNK (1)

    Returns
    -------
    labels : (N,) int64
    """
    n_nodes = int(segments.max()) + 1
    labels  = np.ones(n_nodes, dtype=np.int64)   # default: unknown

    for nid in range(n_nodes):
        m = segments == nid
        if not m.any():
            continue
        fg_ratio = gt_mask[m].mean()
        if fg_ratio >= fg_threshold:
            labels[nid] = CLASS_FG
        elif fg_ratio <= 1 - bg_threshold:
            labels[nid] = CLASS_BG

    return labels


# -----------------------------------------------------------------------
# Processed sample builder
# -----------------------------------------------------------------------

def prepare_sample(
    sample:    dict,
    sp_config: Optional[SuperpixelGraphConfig] = None,
) -> tuple:
    """
    Convert a raw sample dict → (PyG Data, labels tensor, segments array).

    Parameters
    ----------
    sample : dict with keys image, gt_mask, fg_points, bg_points
    sp_config : superpixel configuration

    Returns
    -------
    (data, labels_tensor, segments)
    """
    builder = GraphBuilder(sample["image"], sp_config)
    graph   = builder.build()

    hints   = encode_user_hints(
        graph.segments,
        sample.get("fg_points", []),
        sample.get("bg_points", []),
    )

    x = np.concatenate([graph.node_features, hints], axis=1)   # (N, 19)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_attr,  dtype=torch.float32),
    )

    labels = derive_trimap_labels(graph.segments, sample["gt_mask"])
    return data, torch.tensor(labels, dtype=torch.long), graph.segments


# -----------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------

def load_image_mask_dataset(
    images_dir:     str | Path,
    masks_dir:      str | Path,
    max_size:       int   = 512,
    n_fg_clicks:    int   = 5,
    n_bg_clicks:    int   = 5,
    augment:        bool  = True,
    augment_factor: int   = 2,       # how many augmented copies per original
    click_jitter:   float = 0.01,
) -> list[dict]:
    """
    Load all image/mask pairs from two directories.

    Parameters
    ----------
    images_dir / masks_dir : directories containing images and binary masks.
        Mask filenames must match image filenames (same stem, any extension).
    max_size : resize longest edge to this value.
    augment_factor : each image is replicated this many times with augmentation.

    Returns
    -------
    list of sample dicts
    """
    images_dir = Path(images_dir)
    masks_dir  = Path(masks_dir)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([f for f in images_dir.iterdir()
                          if f.suffix.lower() in image_exts])
    samples = []
    skipped = 0

    for img_path in image_files:
        # Find matching mask
        mask_path = None
        for ext in [".png", ".jpg", ".bmp", ".tif"]:
            c = masks_dir / (img_path.stem + ext)
            if c.exists():
                mask_path = c
                break
        if mask_path is None:
            logger.debug(f"No mask for {img_path.name}, skipping.")
            skipped += 1
            continue

        image   = cv2.imread(str(img_path))
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask_raw is None:
            skipped += 1
            continue

        image, mask_raw = _resize_pair(image, mask_raw, max_size)
        gt_mask = (mask_raw > 127).astype(np.uint8)

        if gt_mask.sum() < 200 or (1 - gt_mask).sum() < 200:
            skipped += 1
            continue

        # Base sample (no augmentation)
        fg_pts, bg_pts = sample_clicks(gt_mask, n_fg_clicks, n_bg_clicks,
                                       jitter=click_jitter)
        if not fg_pts or not bg_pts:
            skipped += 1
            continue

        samples.append({
            "image":     image,
            "gt_mask":   gt_mask,
            "fg_points": fg_pts,
            "bg_points": bg_pts,
            "name":      img_path.stem,
        })

        # Augmented copies
        if augment:
            for aug_i in range(augment_factor):
                aug_img, aug_mask = augment_sample(image, gt_mask)
                fg_aug, bg_aug    = sample_clicks(aug_mask, n_fg_clicks, n_bg_clicks,
                                                  jitter=click_jitter * 2)
                if fg_aug and bg_aug:
                    samples.append({
                        "image":     aug_img,
                        "gt_mask":   aug_mask,
                        "fg_points": fg_aug,
                        "bg_points": bg_aug,
                        "name":      f"{img_path.stem}_aug{aug_i}",
                    })

    logger.info(f"Loaded {len(samples)} samples ({skipped} skipped) from {images_dir}")
    print(f"[Dataset] {len(samples)} samples loaded ({skipped} skipped).")
    return samples


def make_synthetic_dataset(
    n:    int = 200,
    size: int = 128,
    seed: int = 42,
) -> list[dict]:
    """
    Generate synthetic training samples with geometric shapes.

    Useful for:
    - Verifying the pipeline before collecting real data
    - Quick smoke tests
    - Curriculum learning (start synthetic, fine-tune on real)

    Shapes: circles, rectangles, ellipses, L-shapes, rings
    """
    rng = np.random.RandomState(seed)
    samples = []

    for i in range(n):
        img  = rng.randint(20, 100, (size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)

        shape = rng.choice(["circle", "rect", "ellipse", "ring", "Lshape"])
        cx    = rng.randint(size // 4, 3 * size // 4)
        cy    = rng.randint(size // 4, 3 * size // 4)
        color = [int(x) for x in rng.randint(120, 240, 3)]

        if shape == "circle":
            r = rng.randint(size // 8, size // 3)
            cv2.circle(img,  (cx, cy), r, color, -1)
            cv2.circle(mask, (cx, cy), r, 1, -1)

        elif shape == "rect":
            w = rng.randint(size // 6, size // 3)
            h = rng.randint(size // 6, size // 3)
            x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
            x2, y2 = min(size-1, cx + w//2), min(size-1, cy + h//2)
            cv2.rectangle(img,  (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)

        elif shape == "ellipse":
            a = rng.randint(size // 8, size // 3)
            b = rng.randint(size // 12, size // 4)
            angle = rng.randint(0, 180)
            cv2.ellipse(img,  (cx, cy), (a, b), angle, 0, 360, color, -1)
            cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 1, -1)

        elif shape == "ring":
            r_out = rng.randint(size // 5, size // 3)
            r_in  = r_out - rng.randint(size // 15, size // 8)
            cv2.circle(img,  (cx, cy), r_out, color, -1)
            cv2.circle(mask, (cx, cy), r_out, 1, -1)
            bg_color = [int(x) for x in rng.randint(20, 100, 3)]
            cv2.circle(img,  (cx, cy), max(r_in, 1), bg_color, -1)
            cv2.circle(mask, (cx, cy), max(r_in, 1), 0, -1)

        else:  # L-shape
            w, h = rng.randint(size//6, size//3), rng.randint(size//6, size//3)
            t    = max(size // 10, 5)
            x1, y1 = max(0, cx - w//2), max(0, cy - h//2)
            x2, y2 = min(size-1, cx + w//2), min(size-1, cy + h//2)
            cv2.rectangle(img,  (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            # Hollow out inner part
            inner_color = [int(x) for x in rng.randint(20, 100, 3)]
            cv2.rectangle(img,  (x1+t, y1+t), (x2-t, y2-t), inner_color, -1)
            cv2.rectangle(mask, (x1+t, y1+t), (x2-t, y2-t), 0, -1)

        # Add perlin-like noise
        noise = rng.randint(-30, 30, img.shape).astype(np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        fg_pts, bg_pts = sample_clicks(mask, n_fg=3, n_bg=3, erosion_radius=4)
        if not fg_pts or not bg_pts:
            continue

        samples.append({
            "image":     img,
            "gt_mask":   mask,
            "fg_points": fg_pts,
            "bg_points": bg_pts,
            "name":      f"synthetic_{i:04d}_{shape}",
        })

    print(f"[Dataset] Generated {len(samples)} synthetic samples.")
    return samples


def split_dataset(
    samples: list[dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Split into train/val/test sets."""
    random.seed(seed)
    data = samples[:]
    random.shuffle(data)
    n = len(data)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    test   = data[:n_test]
    val    = data[n_test:n_test + n_val]
    train  = data[n_test + n_val:]
    print(f"[Dataset] Split → train:{len(train)}  val:{len(val)}  test:{len(test)}")
    return train, val, test


def _resize_pair(image, mask, max_size):
    H, W  = image.shape[:2]
    scale = max_size / max(H, W)
    if scale < 1.0:
        nW = int(W * scale)
        nH = int(H * scale)
        image = cv2.resize(image, (nW, nH), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (nW, nH), interpolation=cv2.INTER_NEAREST)
    return image, mask
