"""
Build the superpixel graphs for a dataset and store them in the cache.

Graph construction is deterministic in the image, so it is a separate step from
training: run this once per dataset and configuration, then train as often as
needed with the cache warm. Keeping it separate also keeps process-level
parallelism away from a CUDA context, which is a poor combination — the trainer
places its model on the device before it would prepare anything, and worker
processes launched from that state are fragile.

    python3 tools/prepare_graphs.py --images DIR --masks DIR --cache DIR \
        --workers 6 --max-size 384 --augment 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gcn_grabcut.dataset import list_image_mask_pairs, prepare_dataset
from src.gcn_grabcut.graph_builder import SuperpixelGraphConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Warm the graph cache")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-size", type=int, default=384)
    ap.add_argument("--superpixels", type=int, default=300)
    ap.add_argument("--augment", type=int, default=0,
                    help="Seeded augmented copies per image")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1,
                    help="Take every n-th sample, to cover a split sparsely")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    samples = list_image_mask_pairs(args.images, args.masks,
                                    max_size=args.max_size,
                                    augment_copies=args.augment, seed=args.seed)
    if args.stride > 1:
        samples = samples[::args.stride]
    if args.limit:
        samples = samples[:args.limit]

    # Results are discarded: the point of the run is the cache it leaves behind,
    # so nothing needs to be held in memory while it works.
    prepare_dataset(
        samples,
        SuperpixelGraphConfig(n_segments=args.superpixels),
        cache_dir=args.cache,
        workers=args.workers,
        keep_segments=False,
        desc="cache: ",
    )


if __name__ == "__main__":
    main()
