"""
train.py — Train a GCN-GrabCut trimap predictor.

Graphs are deterministic in the image, so they are built once (in parallel,
with an optional on-disk cache) and then reused by every epoch. Training runs
over mini-batches of graphs and selects the checkpoint by validation IoU.

Usage
-----
    python3 train.py --device cuda --epochs 120
    python3 train.py --model gat --batch-size 16 --workers 8
    python3 train.py --cache .graph_cache        # reuse graphs across runs

The module is import-safe: everything runs inside `main()` under a
`__main__` guard, which is required because graph preparation spawns worker
processes and a spawned child imports this file.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from src.gcn_grabcut.dataset import list_image_mask_pairs
from src.gcn_grabcut.graph_builder import SuperpixelGraphConfig
from src.gcn_grabcut.model import build_model
from src.gcn_grabcut.trainer import Trainer, TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GCN-GrabCut")
    parser.add_argument("--images_train", default="data/bsds500/images/train")
    parser.add_argument("--masks_train",  default="data/bsds500/masks/train")
    parser.add_argument("--images_val",   default="data/bsds500/images/val")
    parser.add_argument("--masks_val",    default="data/bsds500/masks/val")
    parser.add_argument("--model",        default="resgcn", choices=["resgcn", "gcn", "gat"])
    parser.add_argument("--epochs",       type=int,   default=120)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--batch-size",   type=int,   default=8,
                        help="Graphs per optimisation step")
    parser.add_argument("--hidden",       type=int,   default=128)
    parser.add_argument("--layers",       type=int,   default=6)
    parser.add_argument("--dropout",      type=float, default=0.15)
    parser.add_argument("--loss",         default="trimap",
                        choices=["trimap", "focal", "smooth_ce", "ce"])
    parser.add_argument("--dice-weight",  type=float, default=0.5)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--checkpoints",  default="checkpoints")
    parser.add_argument("--augment",      type=int,   default=3,
                        help="Augmented copies per training image")
    parser.add_argument("--max-size",     type=int,   default=480)
    parser.add_argument("--superpixels",  type=int,   default=300)
    parser.add_argument("--workers",      type=int,   default=0,
                        help="Processes used to build graphs (0 = serial)")
    parser.add_argument("--cache",        default=None,
                        help="Directory for the persistent graph cache")
    parser.add_argument("--train-limit",  type=int, default=0,
                        help="Cap on training samples (0 = all)")
    parser.add_argument("--val-limit",    type=int, default=0,
                        help="Cap on validation samples (0 = all). A few hundred "
                             "are enough to rank checkpoints, and preparing them "
                             "costs memory that the training set needs")
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    Path(args.checkpoints).mkdir(parents=True, exist_ok=True)
    SP_CFG = SuperpixelGraphConfig(n_segments=args.superpixels)

    print(f"[train] device={args.device} model={args.model} epochs={args.epochs} "
          f"lr={args.lr} batch={args.batch_size}")

    # Datasets are enumerated as descriptors: the parent process holds file names,
    # and each graph builder decodes the single image it works on. Augmented
    # variants are seeded, so they are reproducible and cacheable like originals.
    train_set = list_image_mask_pairs(
        args.images_train, args.masks_train,
        max_size=args.max_size, augment_copies=args.augment, seed=args.seed)
    val_set = list_image_mask_pairs(
        args.images_val, args.masks_val, max_size=args.max_size)

    if args.train_limit:
        train_set = train_set[:args.train_limit]
    if args.val_limit:
        # Evenly spaced rather than the first N, so the subset spans the split.
        step = max(1, len(val_set) // args.val_limit)
        val_set = val_set[::step][:args.val_limit]
    print(f"[train] {len(train_set)} training samples, {len(val_set)} validation")

    model = build_model(
        args.model,
        hidden_channels=args.hidden,
        n_layers=args.layers,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] {model.__class__.__name__} params={n_params:,}")

    cfg = TrainConfig(
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        loss_fn=args.loss,
        dice_weight=args.dice_weight,
        weight_decay=3e-4,
        scheduler="cosine_warm",
        t0=max(args.epochs // 3, 10),
        early_stop_patience=30,
        prep_workers=args.workers,
        cache_dir=args.cache,
        amp=(args.device == "cuda"),
    )

    trainer = Trainer(model, cfg, device=args.device, save_dir=args.checkpoints)
    history = trainer.fit(train_set, val_set, sp_config=SP_CFG)

    if hasattr(model, "layer_weights"):
        weights = np.round(model.layer_weights(), 4).tolist()
        history["fusion_weights"] = weights
        print(f"[train] fusion weights [input, blocks..., sage] = {weights}")

    with open(Path(args.checkpoints) / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    best = max(history["val_score"]) if history.get("val_score") else float("nan")
    print(f"[train] done | best val score = {best:.4f}  (½·(IoU_fg + IoU_bg))")
    print(f"[train] checkpoints -> {args.checkpoints}/")


if __name__ == "__main__":
    main()
