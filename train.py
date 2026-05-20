import argparse
import time
import json
import random
import numpy as np
import torch
from torch.amp import GradScaler, autocast
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.gcn_grabcut import (
    load_image_mask_dataset,
    ResGCNNet, GCNTrimapNet, GATTrimapNet,
    sample_clicks, augment_sample, derive_trimap_labels,
)
from src.gcn_grabcut.graph_builder import GraphBuilder, SuperpixelGraphConfig, encode_user_hints
from src.gcn_grabcut.losses import FocalLoss
from src.gcn_grabcut.model import N_NODE_FEATS
from torch_geometric.data import Data


parser = argparse.ArgumentParser()
parser.add_argument("--images_train", default="data/bsds500/images/train")
parser.add_argument("--masks_train",  default="data/bsds500/masks/train")
parser.add_argument("--images_val",   default="data/bsds500/images/val")
parser.add_argument("--masks_val",    default="data/bsds500/masks/val")
parser.add_argument("--model",        default="resgcn", choices=["resgcn", "gcn", "gat"])
parser.add_argument("--epochs",       type=int,   default=120)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--device",       default="cpu")
parser.add_argument("--checkpoints",  default="checkpoints")
parser.add_argument("--augment",      type=int,   default=3)
args = parser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    args.device = "cpu"
if args.device == "mps" and not torch.backends.mps.is_available():
    args.device = "cpu"

DEVICE = args.device
Path(args.checkpoints).mkdir(exist_ok=True)
print(f"[train] device={DEVICE}  model={args.model}  epochs={args.epochs}  lr={args.lr}")


SP_CFG      = SuperpixelGraphConfig(n_segments=300, compactness=10.0, sigma=1.0)
N_VAL_TRIALS = 3  

def build_graphs_from_raw(raw_samples, augment_factor=0, desc=""):
    records = []
    for s in raw_samples:
        image, mask = s["image"], s["gt_mask"]
        try:
            graph  = GraphBuilder(image, SP_CFG).build()
            labels = derive_trimap_labels(
                graph.segments, mask, fg_threshold=0.70, bg_threshold=0.70
            )
            records.append({
                "node_feats": graph.node_features,
                "edge_index": graph.edge_index,
                "edge_attr":  graph.edge_attr,
                "segments":   graph.segments,
                "gt_mask":    mask,
                "labels":     torch.tensor(labels, dtype=torch.long),
                "image":      image,
            })
        except Exception:
            continue

        for _ in range(augment_factor):
            try:
                aug_img, aug_mask = augment_sample(
                    image, mask,
                    prob_flip=0.5, prob_rotate=0.4,
                    prob_color=0.6, prob_crop=0.4,
                )
                aug_graph  = GraphBuilder(aug_img, SP_CFG).build()
                aug_labels = derive_trimap_labels(
                    aug_graph.segments, aug_mask, fg_threshold=0.70, bg_threshold=0.70
                )
                records.append({
                    "node_feats": aug_graph.node_features,
                    "edge_index": aug_graph.edge_index,
                    "edge_attr":  aug_graph.edge_attr,
                    "segments":   aug_graph.segments,
                    "gt_mask":    aug_mask,
                    "labels":     torch.tensor(aug_labels, dtype=torch.long),
                    "image":      aug_img,
                })
            except Exception:
                continue

    print(f"[train] {desc}{len(records)} graphs built")
    return records


def make_pyg(rec, n_fg, n_bg, jitter):
    fg_pts, bg_pts = sample_clicks(
        rec["gt_mask"], n_fg=n_fg, n_bg=n_bg,
        erosion_radius=6, jitter=jitter,
    )
    hints = encode_user_hints(rec["segments"], fg_pts, bg_pts)
    x = np.concatenate([rec["node_feats"], hints], axis=1)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(rec["edge_attr"],   dtype=torch.float32),
    )


def make_val_ensemble(rec):
    """Multiple click samples per val image for robust evaluation."""
    graphs = [make_pyg(rec, n_fg=5, n_bg=5, jitter=0.01) for _ in range(N_VAL_TRIALS)]
    return graphs, rec["labels"]


print("[train] loading raw images...")
train_raw = load_image_mask_dataset(
    args.images_train, args.masks_train, augment=False, max_size=480)
val_raw = load_image_mask_dataset(
    args.images_val, args.masks_val, augment=False, max_size=480)

print("[train] building train graphs (clicks regenerated each epoch)...")
t0 = time.time()
train_recs = build_graphs_from_raw(train_raw, augment_factor=args.augment, desc="train: ")
print(f"[train] graph build took {time.time()-t0:.1f}s")

print("[train] building val graphs...")
val_recs = build_graphs_from_raw(val_raw, augment_factor=0, desc="val: ")
val_data = [make_val_ensemble(r) for r in val_recs]

model_cls = {"resgcn": ResGCNNet, "gcn": GCNTrimapNet, "gat": GATTrimapNet}[args.model]
model     = model_cls(dropout=0.30).to(DEVICE)
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[train] {model.__class__.__name__}  params={n_params:,}  dropout=0.30")

class_w   = torch.tensor([1.4, 0.7, 1.4], dtype=torch.float32).to(DEVICE)
criterion = FocalLoss(gamma=2.0, weight=class_w)

if hasattr(model, "param_groups"):
    optimizer = AdamW(model.param_groups(args.lr), weight_decay=3e-4)
else:
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=3e-4)

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
scaler    = GradScaler("cuda", enabled=(DEVICE == "cuda"))


def current_lr():
    return optimizer.param_groups[-1]["lr"]


def iou_per_class(preds, gts, n=3):
    ious = []
    for c in range(n):
        tp = ((preds == c) & (gts == c)).sum()
        fp = ((preds == c) & (gts != c)).sum()
        fn = ((preds != c) & (gts == c)).sum()
        ious.append(float(tp) / (float(tp + fp + fn) + 1e-8))
    return ious

def train_epoch():
    model.train()
    total_loss = 0.0
    random.shuffle(train_recs)

    for rec in train_recs:
        n_fg   = random.randint(3, 9)
        n_bg   = random.randint(3, 9)
        jitter = random.uniform(0.0, 0.02)

        try:
            data = make_pyg(rec, n_fg, n_bg, jitter)
        except Exception:
            continue

        data   = data.to(DEVICE)
        labels = rec["labels"].to(DEVICE)

        optimizer.zero_grad()
        with autocast("cuda", enabled=(DEVICE == "cuda")):
            loss = criterion(model(data), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    scheduler.step()
    return total_loss / max(len(train_recs), 1)


@torch.no_grad()
def eval_epoch():
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for graphs, labels in val_data:
        labels = labels.to(DEVICE)

        logits_sum = None
        for data in graphs:
            data   = data.to(DEVICE)
            logits = model(data)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits_avg = logits_sum / len(graphs)

        total_loss += criterion(logits_avg, labels).item()
        all_preds.append(logits_avg.argmax(-1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    gts   = torch.cat(all_labels).numpy()
    ious  = iou_per_class(preds, gts)
    return {
        "loss":    total_loss / max(len(val_data), 1),
        "acc":     float((preds == gts).mean()),
        "iou_bg":  ious[0],
        "iou_unk": ious[1],
        "iou_fg":  ious[2],
    }


best_iou_fg = 0.0
patience    = 0
PATIENCE    = 30
history     = {"train_loss": [], "val_loss": [], "val_acc": [], "iou_fg": [], "lr": []}

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    tl = train_epoch()
    vm = eval_epoch()

    history["train_loss"].append(tl)
    history["val_loss"].append(vm["loss"])
    history["val_acc"].append(vm["acc"])
    history["iou_fg"].append(vm["iou_fg"])
    history["lr"].append(current_lr())

    if vm["iou_fg"] > best_iou_fg:
        best_iou_fg = vm["iou_fg"]
        patience    = 0
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "iou_fg": best_iou_fg},
            f"{args.checkpoints}/best_model.pt",
        )
    else:
        patience += 1

    if epoch % 5 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train={tl:.4f} | val={vm['loss']:.4f} | "
            f"acc={vm['acc']:.4f} | IoU_fg={vm['iou_fg']:.4f} | "
            f"lr={current_lr():.2e} | {time.time()-t0:.1f}s"
        )

    if patience >= PATIENCE:
        print(f"[train] early stopping at epoch {epoch} (patience={PATIENCE})")
        break

torch.save({"model": model.state_dict(), "epoch": epoch},
           f"{args.checkpoints}/final_model.pt")

with open(f"{args.checkpoints}/history.json", "w") as f:
    json.dump(history, f, indent=2)

print(f"\n[train] done  |  best IoU_fg: {best_iou_fg:.4f}")
print(f"[train] checkpoints → {args.checkpoints}/")