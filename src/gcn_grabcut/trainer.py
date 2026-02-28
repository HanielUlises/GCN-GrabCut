"""
Training engine for GCN-GrabCut trimap prediction models.

Features
--------
- Mixed-precision training (FP16) when CUDA available
- Cosine annealing LR with warm restarts
- Early stopping with patience
- Per-epoch metrics: loss, acc, IoU-per-class
- TensorBoard logging (optional)
- Checkpoint save/load with full training state
- Class-balanced cross-entropy loss
- Focal loss option for hard negative mining
"""

from __future__ import annotations

import numpy as np
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW, SGD
    from torch.optim.lr_scheduler import (
        CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
    )
    from torch.cuda.amp import GradScaler, autocast
    _TORCH = True
except ImportError:
    _TORCH = False

from .dataset import prepare_sample, split_dataset
from .graph_builder import SuperpixelGraphConfig
from .model import CLASS_BG, CLASS_UNK, CLASS_FG

logger = logging.getLogger(__name__)

if _TORCH:

    class FocalLoss(nn.Module):
        """
        Focal Loss for class imbalance.
        FL(p) = -α(1-p)^γ · log(p)

        Especially useful when UNKNOWN class dominates the label distribution.
        """
        def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
            super().__init__()
            self.gamma  = gamma
            self.weight = weight

        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            ce    = F.cross_entropy(logits, labels, weight=self.weight, reduction="none")
            p_t   = torch.exp(-ce)
            focal = ((1 - p_t) ** self.gamma) * ce
            return focal.mean()


    class LabelSmoothingCE(nn.Module):
        """Cross-entropy with label smoothing — reduces overconfidence."""
        def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None):
            super().__init__()
            self.smoothing = smoothing
            self.weight    = weight

        def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            n_classes = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)

            with torch.no_grad():
                smooth = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
                smooth.scatter_(1, labels.unsqueeze(1), 1.0 - self.smoothing)
            loss = -(smooth * log_probs).sum(dim=-1)
            if self.weight is not None:
                w = self.weight[labels]
                loss = loss * w
            return loss.mean()


    @dataclass
    class TrainConfig:
        n_epochs:        int   = 60
        lr:              float = 1e-3
        weight_decay:    float = 1e-4
        optimizer:       str   = "adamw"
        scheduler:       str   = "cosine_warm"
        loss_fn:         str   = "focal"
        focal_gamma:     float = 2.0
        label_smoothing: float = 0.1
        class_weights:   list  = field(default_factory=lambda: [1.5, 0.8, 1.5])
        amp:             bool  = True
        grad_clip:       float = 1.0
        early_stop_patience: int = 15
        t0:              int   = 10
        t_mult:          int   = 2
        val_every:       int   = 1
        save_every:      int   = 5
        verbose:         bool  = True
        log_dir:         Optional[str] = None

    class Trainer:
        """
        Full training engine for GCNTrimapPredictor / ResGCNNet / GATTrimapNet.

        Parameters
        ----------
        model   : the GCN model to train
        config  : TrainConfig dataclass
        device  : "cuda" | "cpu" | "mps"
        save_dir: directory for checkpoints
        """

        def __init__(
            self,
            model:    nn.Module,
            config:   Optional[TrainConfig] = None,
            device:   str  = "cpu",
            save_dir: str  = "checkpoints",

            lr:             Optional[float] = None,
            n_epochs:       Optional[int]   = None,
            class_weights:  Optional[Sequence[float]] = None,
        ):
            self.cfg      = config or TrainConfig()

            if lr is not None:          self.cfg.lr       = lr
            if n_epochs is not None:    self.cfg.n_epochs = n_epochs
            if class_weights is not None: self.cfg.class_weights = list(class_weights)

            self.device   = device
            self.model    = model.to(device)
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)

            w = torch.tensor(self.cfg.class_weights, dtype=torch.float32).to(device) \
                if self.cfg.class_weights else None

            if self.cfg.loss_fn == "focal":
                self.criterion = FocalLoss(gamma=self.cfg.focal_gamma, weight=w)
            elif self.cfg.loss_fn == "smooth_ce":
                self.criterion = LabelSmoothingCE(
                    smoothing=self.cfg.label_smoothing, weight=w
                )
            else:
                self.criterion = nn.CrossEntropyLoss(weight=w)


            if hasattr(model, "param_groups"):
                param_groups = model.param_groups(self.cfg.lr)
            else:
                param_groups = [{"params": model.parameters(), "lr": self.cfg.lr}]

            if self.cfg.optimizer == "sgd":
                self.optimizer = SGD(
                    param_groups, lr=self.cfg.lr,
                    momentum=0.9, weight_decay=self.cfg.weight_decay, nesterov=True
                )
            else:
                self.optimizer = AdamW(
                    param_groups, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
                )


            self.scheduler = None


            self.scaler = GradScaler() if (self.cfg.amp and device == "cuda") else None


            self.history = {
                "train_loss": [], "val_loss": [],
                "val_acc":    [], "val_iou_bg": [], "val_iou_unk": [], "val_iou_fg": [],
                "lr":         [],
            }
            self._best_val_loss = float("inf")
            self._patience_ctr  = 0

            self._tb = None
            if self.cfg.log_dir:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._tb = SummaryWriter(self.cfg.log_dir)
                except ImportError:
                    logger.warning("tensorboard not installed; skipping TB logging.")


        def fit(
            self,
            train_samples: list[dict],
            val_samples:   Optional[list[dict]] = None,
            sp_config:     Optional[SuperpixelGraphConfig] = None,
        ) -> dict:
            """
            Train the model end-to-end.

            Parameters
            ----------
            train_samples : list of raw sample dicts
            val_samples   : validation split (or None)
            sp_config     : superpixel config; uses default if None

            Returns
            -------
            history dict with per-epoch metrics
            """
            print(f"[Trainer] Preparing graphs for {len(train_samples)} train samples...")
            t_start = time.time()
            train_data = [prepare_sample(s, sp_config) for s in train_samples]
            val_data   = [prepare_sample(s, sp_config) for s in val_samples] \
                         if val_samples else None
            print(f"[Trainer] Graph preparation took {time.time()-t_start:.1f}s")

            self._init_scheduler(len(train_data))

            cfg = self.cfg
            for epoch in range(1, cfg.n_epochs + 1):
                t0  = time.time()
                tl  = self._train_epoch(train_data)
                self.history["train_loss"].append(tl)
                self.history["lr"].append(self._current_lr())

                if val_data and epoch % cfg.val_every == 0:
                    vm = self._eval_epoch(val_data)
                    self.history["val_loss"].append(vm["loss"])
                    self.history["val_acc"].append(vm["acc"])
                    self.history["val_iou_bg"].append(vm["iou_bg"])
                    self.history["val_iou_unk"].append(vm["iou_unk"])
                    self.history["val_iou_fg"].append(vm["iou_fg"])

                    if self._tb:
                        self._tb.add_scalar("val/loss",    vm["loss"],    epoch)
                        self._tb.add_scalar("val/acc",     vm["acc"],     epoch)
                        self._tb.add_scalar("val/iou_fg",  vm["iou_fg"],  epoch)

                    if vm["loss"] < self._best_val_loss:
                        self._best_val_loss = vm["loss"]
                        self._patience_ctr  = 0
                        self._save("best_model.pt", epoch=epoch, val_loss=vm["loss"])
                    else:
                        self._patience_ctr += 1

                    if cfg.verbose and epoch % 5 == 0:
                        dt = time.time() - t0
                        print(
                            f"Epoch {epoch:3d}/{cfg.n_epochs} | "
                            f"train_loss={tl:.4f} | val_loss={vm['loss']:.4f} | "
                            f"val_acc={vm['acc']:.4f} | IoU_fg={vm['iou_fg']:.4f} | "
                            f"lr={self._current_lr():.2e} | {dt:.1f}s"
                        )

                    if self._patience_ctr >= cfg.early_stop_patience:
                        print(f"[Trainer] Early stopping at epoch {epoch} "
                              f"(no improvement for {cfg.early_stop_patience} epochs).")
                        break
                else:
                    if cfg.verbose and epoch % 5 == 0:
                        print(f"Epoch {epoch:3d}/{cfg.n_epochs} | "
                              f"train_loss={tl:.4f} | lr={self._current_lr():.2e}")

                if self._tb:
                    self._tb.add_scalar("train/loss", tl, epoch)
                    self._tb.add_scalar("train/lr",   self._current_lr(), epoch)

                if epoch % cfg.save_every == 0:
                    self._save(f"epoch_{epoch:04d}.pt", epoch=epoch, val_loss=None)

            self._save("final_model.pt", epoch=cfg.n_epochs, val_loss=None)
            self._save_history()
            if self._tb:
                self._tb.close()
            return self.history

        def _train_epoch(self, data_list: list) -> float:
            self.model.train()
            total_loss = 0.0
            order      = torch.randperm(len(data_list)).tolist()

            for idx in order:
                data, labels, _ = data_list[idx]
                data   = data.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                if self.scaler:
                    with autocast():
                        logits = self.model(data)
                        loss   = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(data)
                    loss   = self.criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()

                total_loss += loss.item()

                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            if self.scheduler and not isinstance(self.scheduler, (OneCycleLR, ReduceLROnPlateau)):
                self.scheduler.step()

            return total_loss / max(len(data_list), 1)

        @torch.no_grad()
        def _eval_epoch(self, data_list: list) -> dict:
            self.model.eval()
            total_loss = 0.0
            all_preds  = []
            all_labels = []

            for data, labels, _ in data_list:
                data   = data.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(data)
                total_loss += self.criterion(logits, labels).item()
                all_preds.append(logits.argmax(dim=-1).cpu())
                all_labels.append(labels.cpu())

            preds  = torch.cat(all_preds)
            gts    = torch.cat(all_labels)
            acc    = (preds == gts).float().mean().item()
            ious   = _per_class_iou(preds.numpy(), gts.numpy(), n_classes=3)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(total_loss / max(len(data_list), 1))

            return {
                "loss":    total_loss / max(len(data_list), 1),
                "acc":     acc,
                "iou_bg":  ious[CLASS_BG],
                "iou_unk": ious[CLASS_UNK],
                "iou_fg":  ious[CLASS_FG],
            }

        def _init_scheduler(self, steps_per_epoch: int) -> None:
            cfg = self.cfg
            if cfg.scheduler == "cosine_warm":
                self.scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=cfg.t0, T_mult=cfg.t_mult
                )
            elif cfg.scheduler == "onecycle":
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=cfg.lr,
                    total_steps=cfg.n_epochs * steps_per_epoch,
                    pct_start=0.1,
                )
            elif cfg.scheduler == "plateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
                )
            else:
                self.scheduler = None

        def _current_lr(self) -> float:
            return self.optimizer.param_groups[-1]["lr"]

        def _save(self, filename: str, epoch: int, val_loss: Optional[float]) -> None:
            path  = self.save_dir / filename
            state = {
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch":     epoch,
                "val_loss":  val_loss,
            }
            if self.scheduler:
                state["scheduler"] = self.scheduler.state_dict()
            torch.save(state, path)
            logger.debug(f"Saved checkpoint → {path}")

        def load(self, filename: str, weights_only: bool = True) -> int:
            """Load checkpoint. Returns the saved epoch number."""
            path  = self.save_dir / filename
            ckpt  = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])
            if not weights_only:
                self.optimizer.load_state_dict(ckpt["optimizer"])
                if self.scheduler and "scheduler" in ckpt:
                    self.scheduler.load_state_dict(ckpt["scheduler"])
            logger.info(f"Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")
            return ckpt.get("epoch", 0)

        def _save_history(self) -> None:
            path = self.save_dir / "history.json"
            with open(path, "w") as f:
                json.dump(self.history, f, indent=2)
            print(f"[Trainer] History saved → {path}")


    def _per_class_iou(preds: np.ndarray, gts: np.ndarray, n_classes: int) -> list[float]:
        ious = []
        for c in range(n_classes):
            tp = ((preds == c) & (gts == c)).sum()
            fp = ((preds == c) & (gts != c)).sum()
            fn = ((preds != c) & (gts == c)).sum()
            ious.append(float(tp / (tp + fp + fn + 1e-8)))
        return ious
