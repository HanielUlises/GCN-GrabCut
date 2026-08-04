"""
Dataset utilities for GCN-GrabCut training.

Responsibilities
----------------
1. Load image/mask pairs from disk.
2. Apply augmentation (flip, colour jitter, random rotation, crop).
3. Pre-compute superpixel graphs + derive per-superpixel trimap labels.
4. Cache processed graphs to speed up subsequent epochs.

Training is click-free: node inputs carry the automatic FG/BG prior computed
by `GraphBuilder`, exactly as at inference time.

Sample dict schema
------------------
{
  "image"     : np.ndarray  (H, W, 3) BGR uint8
  "gt_mask"   : np.ndarray  (H, W)    uint8 binary {0, 1}
  "name"      : str
}

Processed tuple: (PyG Data, labels tensor (N,), segments array (H,W))
"""

from __future__ import annotations

import numpy as np
import cv2
import os
import random
import time
import zlib
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

from .graph_builder import GraphBuilder, SuperpixelGraphConfig
from .model import CLASS_BG, CLASS_UNK, CLASS_FG, N_NODE_FEATS


# -----------------------------------------------------------------------
# Click simulation (legacy — unused by the automatic pipeline, kept for
# ablations against the old interactive baseline)
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
    flat    = segments.ravel()

    counts = np.bincount(flat, minlength=n_nodes).astype(np.float64)
    fg_sum = np.bincount(flat, weights=(gt_mask.ravel() > 0).astype(np.float64),
                         minlength=n_nodes)
    fg_ratio = fg_sum / np.maximum(counts, 1.0)

    labels = np.full(n_nodes, CLASS_UNK, dtype=np.int64)
    labels[fg_ratio >= fg_threshold]     = CLASS_FG
    labels[fg_ratio <= 1 - bg_threshold] = CLASS_BG
    labels[counts == 0]                  = CLASS_UNK
    return labels


# -----------------------------------------------------------------------
# Processed sample builder
# -----------------------------------------------------------------------

def prepare_sample(
    sample:    dict,
    sp_config: Optional[SuperpixelGraphConfig] = None,
    fg_threshold: float = 0.70,
    bg_threshold: float = 0.70,
) -> tuple:
    """
    Convert a raw sample dict → (PyG Data, labels tensor, segments array).

    The returned graph carries three supervision-relevant tensors besides the
    node/edge features: `node_area` (region size as a fraction of the image),
    `y` (per-region trimap label) and `fg_ratio` (soft foreground coverage).
    Area weighting lets the loss approximate a pixel-level objective while
    still being evaluated once per region, and the soft ratio is what makes
    boundary regions contribute a graded rather than binary signal.

    Parameters
    ----------
    sample : dict with keys image, gt_mask
    sp_config : superpixel configuration
    fg_threshold / bg_threshold : region purity required for a definite label

    Returns
    -------
    (data, labels_tensor, segments)
    """
    builder = GraphBuilder(sample["image"], sp_config)
    graph   = builder.build()

    segments = graph.segments
    gt_mask  = sample["gt_mask"]
    n_nodes  = graph.n_nodes
    flat     = segments.ravel()
    counts   = np.bincount(flat, minlength=n_nodes).astype(np.float32)
    fg_ratio = (np.bincount(flat, weights=(gt_mask.ravel() > 0).astype(np.float64),
                            minlength=n_nodes) / np.maximum(counts, 1.0)).astype(np.float32)

    labels = derive_trimap_labels(segments, gt_mask, fg_threshold, bg_threshold)

    data = Data(
        x=torch.tensor(graph.node_input(), dtype=torch.float32),   # (N, 19)
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_attr,  dtype=torch.float32),
        node_area=torch.tensor(graph.node_areas, dtype=torch.float32),
        fg_ratio=torch.tensor(fg_ratio, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
    )
    return data, data.y, segments


def list_image_mask_pairs(
    images_dir:     str | Path,
    masks_dir:      str | Path,
    max_size:       int = 512,
    augment_copies: int = 0,
    seed:           int = 0,
) -> list[dict]:
    """
    Enumerate image/mask pairs as *descriptors* rather than decoded pixels.

    A descriptor names the files, the resize target, and — for augmented
    variants — the seed that determines the transform. Nothing is read from
    disk here, so a dataset of any size costs kilobytes in the parent process,
    and each graph builder decodes only the one image it is working on. This
    matters as soon as a dataset no longer fits in memory: holding ten thousand
    decoded images and then sending copies of them to a process pool is what
    makes preparation run out of memory rather than out of time.

    Augmented variants are seeded, so the same descriptor always yields the
    same transformed image and can therefore be cached like any other.

    Returns
    -------
    list of dicts with keys image_path, mask_path, name, max_size, aug_seed
    """
    images_dir, masks_dir = Path(images_dir), Path(masks_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    out, missing = [], 0
    for img_path in sorted(f for f in images_dir.iterdir()
                           if f.suffix.lower() in image_exts):
        mask_path = next((masks_dir / (img_path.stem + ext)
                          for ext in (".png", ".jpg", ".bmp", ".tif")
                          if (masks_dir / (img_path.stem + ext)).exists()), None)
        if mask_path is None:
            missing += 1
            continue

        base = dict(image_path=str(img_path), mask_path=str(mask_path),
                    max_size=max_size)
        out.append({**base, "name": img_path.stem, "aug_seed": None})
        for k in range(augment_copies):
            # crc32, not hash(): string hashing is salted per interpreter, so the
            # same variant would draw a different seed — and therefore occupy a
            # different cache entry — on every run.
            stem_id = zlib.crc32(img_path.stem.encode()) % 100003
            out.append({**base, "name": f"{img_path.stem}_aug{k}",
                        "aug_seed": seed + 1000003 * k + stem_id})

    print(f"[Dataset] {len(out)} descriptors from {images_dir.name} "
          f"({missing} without a mask)")
    return out


def materialise(sample: dict) -> Optional[dict]:
    """
    Decode a descriptor into an image/mask pair, applying seeded augmentation.

    Samples that already carry pixel arrays are returned unchanged, so both
    descriptor-based and in-memory datasets flow through the same code path.
    Returns None when the pair is unreadable or degenerate.
    """
    if "image" in sample and "gt_mask" in sample:
        return sample

    # A decode is retried before the pair is given up on: under concurrent load
    # from slow or removable storage, reads fail intermittently and OpenCV
    # reports that as None rather than as an error. Treating the first failure
    # as final silently shrinks the dataset, which is worse than being slow.
    image = mask = None
    for attempt in range(3):
        image = cv2.imread(sample["image_path"])
        mask  = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is not None and mask is not None:
            break
        time.sleep(0.05 * (attempt + 1))
    if image is None or mask is None:
        logger.warning("unreadable pair: %s", sample.get("image_path"))
        return None

    image, mask = _resize_pair(image, mask, sample.get("max_size", 512))
    gt_mask = (mask > 127).astype(np.uint8)

    if sample.get("aug_seed") is not None:
        state = random.getstate()
        random.seed(sample["aug_seed"])
        try:
            image, gt_mask = augment_sample(
                image, gt_mask,
                prob_flip=0.5, prob_rotate=0.4, prob_color=0.6, prob_crop=0.4,
            )
        finally:
            random.setstate(state)

    if gt_mask.sum() < 200 or (1 - gt_mask).sum() < 200:
        return None

    return {"image": image, "gt_mask": gt_mask, "name": sample.get("name", "")}


def _cache_key(sample: dict, sp_config: Optional[SuperpixelGraphConfig],
               fg_threshold: float, bg_threshold: float) -> str:
    import hashlib
    cfg = sp_config or SuperpixelGraphConfig()
    h   = hashlib.sha1()
    if "image" in sample:
        h.update(np.ascontiguousarray(sample["image"]))
        h.update(np.ascontiguousarray(sample["gt_mask"]))
    else:
        h.update(repr((sample["image_path"], sample["mask_path"],
                       sample.get("max_size"), sample.get("aug_seed"))).encode())
    h.update(repr((cfg.n_segments, cfg.compactness, cfg.sigma, cfg.use_lab,
                   cfg.connectivity, cfg.n_nonlocal,
                   fg_threshold, bg_threshold)).encode())
    return h.hexdigest()[:20]


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")


def _worker_init() -> None:
    """
    Restrict each worker to one compute thread.

    Every process here handles a single small image, so the parallelism that
    matters is across processes. Left alone, each worker starts a full
    OpenMP/BLAS thread pool sized to the machine; with ten workers that is
    hundreds of threads contending for sixteen cores, and they spend their time
    blocked on one another's futexes rather than building graphs.
    """
    try:
        cv2.setNumThreads(0)
    except Exception:
        pass
    if _TORCH:
        torch.set_num_threads(1)


def _prepare_one(args: tuple):
    """
    Worker entry point for parallel graph preparation.

    `keep_segments` is honoured here rather than in the caller because the
    label map is returned across a process boundary: discarding it before the
    return keeps it from being pickled at all, which for a dataset of tens of
    thousands of images is several gigabytes that never need to move.
    """
    sample, sp_config, fg_t, bg_t, cache_dir, keep_segments = args
    path = None
    if cache_dir is not None:
        path = Path(cache_dir) / f"{_cache_key(sample, sp_config, fg_t, bg_t)}.pt"
        if path.exists():
            try:
                blob = torch.load(path, map_location="cpu", weights_only=False)
                data = blob["data"]
                return data, data.y, (blob["segments"] if keep_segments else None)
            except Exception:
                pass   # corrupt or stale cache entry — rebuild it

    # Decoding happens after the cache lookup, so a cache hit never reads the
    # source image at all.
    sample = materialise(sample)
    if sample is None:
        return None

    data, labels, segments = prepare_sample(sample, sp_config, fg_t, bg_t)
    if path is not None:
        # Written to a temporary name and renamed, so that a run interrupted
        # mid-write cannot leave a truncated entry behind: a partially written
        # archive can take the loading process down rather than merely raise.
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save({"data": data, "segments": segments}, tmp)
            os.replace(tmp, path)
        except Exception:
            tmp.unlink(missing_ok=True)
    return data, labels, (segments if keep_segments else None)


def prepare_dataset(
    samples:        list[dict],
    sp_config:      Optional[SuperpixelGraphConfig] = None,
    fg_threshold:   float = 0.70,
    bg_threshold:   float = 0.70,
    cache_dir:      Optional[str | Path] = None,
    workers:        int = 0,
    desc:           str = "",
    keep_segments:  bool = True,
) -> list[tuple]:
    """
    Build the graph for every sample, in parallel and with an optional cache.

    Graph construction is deterministic in the image, so it is done once and
    reused for every epoch. Superpixel extraction is CPU-bound and releases
    nothing to the GPU, which makes it the natural place to spend processes:
    `workers > 0` fans the work out, `cache_dir` persists the result across
    runs so a second experiment on the same data starts training immediately.

    The pixel label map is two orders of magnitude larger than the graph it
    produced and is not read during optimisation, so `keep_segments=False`
    discards it after use; on a dataset of tens of thousands of images that is
    the difference between a few hundred megabytes and several gigabytes of
    resident memory.

    Returns
    -------
    list of (Data, labels, segments); segments is None when not kept, and
    samples that fail to build are dropped.
    """
    jobs = [(s, sp_config, fg_threshold, bg_threshold,
             str(cache_dir) if cache_dir else None, keep_segments)
            for s in samples]

    records: list[tuple] = []
    failures: list[str] = []
    t0 = time.perf_counter()

    if workers and workers > 1 and len(jobs) > 1:
        # Jobs are submitted individually rather than mapped, so that a failing
        # sample is isolated instead of discarding the whole batch, and the pool
        # is rebuilt if a worker dies outright — over tens of thousands of
        # images a single unreadable file should not cost the entire run.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool

        # Workers are spawned rather than forked. By the time a dataset is
        # prepared the parent has usually initialised a CUDA context — the model
        # is placed on the device when the trainer is constructed — and a forked
        # child inheriting that context is unsupported: the child dies without
        # raising, which surfaces only as a broken pool.
        ctx = multiprocessing.get_context("spawn")

        # Thread limits are exported before the pool is created so that spawned
        # children inherit them at interpreter start, when the numerical
        # libraries size their pools; the parent's own settings are restored
        # afterwards.
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: "1" for k in _THREAD_VARS})

        pending, attempt = list(jobs), 0
        while pending and attempt < 3:
            attempt += 1
            n_workers = max(1, workers // attempt)
            # Work is submitted in bounded windows rather than all at once:
            # queueing tens of thousands of futures against a single pool was
            # observed to take the workers down, while the same jobs complete
            # reliably a few thousand at a time. The window also gives a natural
            # place to report progress on a preparation that runs for minutes.
            window = max(512, n_workers * 128)
            unfinished: list[tuple] = []
            queue = pending

            try:
                with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                         initializer=_worker_init) as pool:
                    for start in range(0, len(queue), window):
                        chunk = queue[start:start + window]
                        futures = {pool.submit(_prepare_one, j): j for j in chunk}
                        try:
                            for fut in as_completed(futures):
                                try:
                                    out = fut.result()
                                except Exception as exc:
                                    failures.append(repr(exc))
                                    continue
                                if out is not None:
                                    records.append(out)
                        except BrokenProcessPool:
                            unfinished = ([j for f, j in futures.items() if not f.done()]
                                          + queue[start + window:])
                            break
                        if len(queue) > window:
                            done = min(start + window, len(queue))
                            print(f"[Dataset] {desc}{done}/{len(queue)} prepared "
                                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
            except BrokenProcessPool:
                unfinished = unfinished or queue
            pending = unfinished
            if pending:
                print(f"[Dataset] worker pool died; retrying {len(pending)} "
                      f"samples with {max(1, workers // (attempt + 1))} workers")

        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    else:
        for job in jobs:
            try:
                out = _prepare_one(job)
                if out is not None:
                    records.append(out)
            except Exception as exc:
                failures.append(repr(exc))

    print(f"[Dataset] {desc}{len(records)}/{len(samples)} graphs ready "
          f"in {time.perf_counter() - t0:.1f}s"
          + (f" (cache: {cache_dir})" if cache_dir else ""))

    # Losing samples silently would misreport the size of the training set, so
    # the tally and a sample of the distinct causes are always printed.
    lost = len(samples) - len(records)
    if lost:
        seen, distinct = set(), []
        for f in failures:
            if f not in seen:
                seen.add(f)
                distinct.append(f)
        # `failures` counts raised futures, which retries can inflate past the
        # number of samples, so the two tallies are reported independently
        # rather than one being derived from the other.
        print(f"[Dataset] {desc}{lost} sample(s) missing from the result; "
              f"{len(failures)} failure(s) raised across attempts")
        for f in distinct[:3]:
            print(f"[Dataset]   {f}")
    return records


# -----------------------------------------------------------------------
# Dataset loaders
# -----------------------------------------------------------------------

def load_image_mask_dataset(
    images_dir:     str | Path,
    masks_dir:      str | Path,
    max_size:       int   = 512,
    augment:        bool  = True,
    augment_factor: int   = 2,       # how many augmented copies per original
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
        samples.append({
            "image":   image,
            "gt_mask": gt_mask,
            "name":    img_path.stem,
        })

        # Augmented copies
        if augment:
            for aug_i in range(augment_factor):
                aug_img, aug_mask = augment_sample(image, gt_mask)
                samples.append({
                    "image":   aug_img,
                    "gt_mask": aug_mask,
                    "name":    f"{img_path.stem}_aug{aug_i}",
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

        if mask.sum() == 0 or (1 - mask).sum() == 0:
            continue

        samples.append({
            "image":   img,
            "gt_mask": mask,
            "name":    f"synthetic_{i:04d}_{shape}",
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
