import argparse
import torch
from pathlib import Path

from src.gcn_grabcut import (
    load_image_mask_dataset,
    split_dataset,
    Trainer,
    TrainConfig,
    ResGCNNet,
    GCNTrimapNet,
    GATTrimapNet,
)

parser = argparse.ArgumentParser()
parser.add_argument("--images_train", default="data/bsds500/images/train")
parser.add_argument("--masks_train",  default="data/bsds500/masks/train")
parser.add_argument("--images_val",   default="data/bsds500/images/val")
parser.add_argument("--masks_val",    default="data/bsds500/masks/val")
parser.add_argument("--model",        default="resgcn", choices=["resgcn", "gcn", "gat"])
parser.add_argument("--epochs",       type=int,   default=60)
parser.add_argument("--lr",           type=float, default=1e-3)
parser.add_argument("--device",       default="cpu", help="cpu | cuda | mps")
parser.add_argument("--checkpoints",  default="checkpoints")
parser.add_argument("--augment",      type=int, default=2, help="augmentation copies per image")
args = parser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    print("[warn] CUDA not available, falling back to CPU")
    args.device = "cpu"
if args.device == "mps" and not torch.backends.mps.is_available():
    print("[warn] MPS not available, falling back to CPU")
    args.device = "cpu"

print(f"[train] device={args.device}  model={args.model}  epochs={args.epochs}  lr={args.lr}")

print("[train] loading train set...")
train_samples = load_image_mask_dataset(
    args.images_train,
    args.masks_train,
    augment=True,
    augment_factor=args.augment,
)
print(f"[train] {len(train_samples)} train samples loaded")

print("[train] loading val set...")
val_samples = load_image_mask_dataset(
    args.images_val,
    args.masks_val,
    augment=False,
)
print(f"[train] {len(val_samples)} val samples loaded")

models = {
    "resgcn": ResGCNNet,
    "gcn":    GCNTrimapNet,
    "gat":    GATTrimapNet,
}
model = models[args.model]()
print(f"[train] model: {model.__class__.__name__}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[train] trainable params: {n_params:,}")

Path(args.checkpoints).mkdir(exist_ok=True)

cfg = TrainConfig(
    n_epochs=args.epochs,
    lr=args.lr,
    amp=(args.device == "cuda"),
)

trainer = Trainer(
    model=model,
    config=cfg,
    device=args.device,
    save_dir=args.checkpoints,
)

history = trainer.fit(train_samples, val_samples)

print("[train] finished!")
print(f"[train] best val loss: {min(history['val_loss']):.4f}" if history["val_loss"] else "")
print(f"[train] checkpoints saved to: {args.checkpoints}/")