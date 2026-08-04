"""
Training engine for GCN-GrabCut trimap prediction models.

Features
--------
- Graphs are built once, in parallel, and optionally cached on disk
- Mini-batch training over graphs (PyG `Batch`), not one graph per step
- Mixed-precision training when CUDA is available
- Area-weighted focal + soft-Dice objective, or plain focal / smoothed CE
- Cosine annealing LR with warm restarts, one-cycle, or plateau decay
- Model selection and early stopping on validation IoU
- Checkpoint save/load with full training state, TensorBoard logging (optional)
"""

from __future__ import annotations

import numpy as np
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence
from .losses import FocalLoss, LabelSmoothingCE, TrimapLoss

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import AdamW, SGD
    from torch.optim.lr_scheduler import (
        CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau
    )
    from torch.amp import GradScaler, autocast
    from torch_geometric.data import Batch
    _TORCH = True
except ImportError:
    _TORCH = False

from .dataset import prepare_dataset, split_dataset
from .graph_builder import SuperpixelGraphConfig
from .model import CLASS_BG, CLASS_UNK, CLASS_FG

logger = logging.getLogger(__name__)

if _TORCH:

    @dataclass
    class TrainConfig:
        n_epochs:        int   = 60
        lr:              float = 1e-3
        weight_decay:    float = 1e-4
        optimizer:       str   = "adamw"
        scheduler:       str   = "cosine_warm"
        loss_fn:         str   = "trimap"
        focal_gamma:     float = 2.0
        dice_weight:     float = 0.5
        label_smoothing: float = 0.1
        class_weights:   list  = field(default_factory=lambda: [1.5, 0.8, 1.5])
        batch_size:      int   = 8
        amp:             bool  = True
        grad_clip:       float = 1.0
        early_stop_patience: int = 15
        t0:              int   = 10
        t_mult:          int   = 2
        val_every:       int   = 1
        save_every:      int   = 5
        prep_workers:    int   = 0
        cache_dir:       Optional[str] = None
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

            if self.cfg.loss_fn == "trimap":
                self.criterion = TrimapLoss(
                    gamma=self.cfg.focal_gamma, weight=w,
                    dice_weight=self.cfg.dice_weight,
                )
            elif self.cfg.loss_fn == "focal":
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


            self.scaler = GradScaler("cuda") if (self.cfg.amp and device == "cuda") else None


            self.history = {
                "train_loss": [], "val_loss": [],
                "val_acc":    [], "val_iou_bg": [], "val_iou_unk": [], "val_iou_fg": [],
                "val_score":  [], "lr": [],
            }
            self._best_score   = -float("inf")
            self._patience_ctr = 0

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
            cfg = self.cfg
            train_data = prepare_dataset(
                train_samples, sp_config, cache_dir=cfg.cache_dir,
                workers=cfg.prep_workers, desc="train: ", keep_segments=False,
            )
            val_data = prepare_dataset(
                val_samples, sp_config, cache_dir=cfg.cache_dir,
                workers=cfg.prep_workers, desc="val: ", keep_segments=False,
            ) if val_samples else None

            # An empty split would otherwise "train" silently: every epoch
            # would average zero batches, report a loss of zero, and save
            # checkpoints that were never updated.
            if not train_data:
                raise RuntimeError(
                    f"no training graphs were prepared from {len(train_samples)} "
                    "samples — check the image and mask directories, and the "
                    "preparation warnings above")
            if val_samples and not val_data:
                raise RuntimeError(
                    f"no validation graphs were prepared from {len(val_samples)} "
                    "samples; model selection would have nothing to rank")

            self._init_scheduler(self._n_steps(len(train_data)))

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

                    self.history["val_score"].append(vm["score"])
                    if vm["score"] > self._best_score:
                        self._best_score   = vm["score"]
                        self._patience_ctr = 0
                        self._save("best_model.pt", epoch=epoch,
                                   val_loss=vm["loss"], score=vm["score"])
                    else:
                        self._patience_ctr += 1

                    if cfg.verbose and epoch % 5 == 0:
                        dt = time.time() - t0
                        print(
                            f"Epoch {epoch:3d}/{cfg.n_epochs} | "
                            f"train_loss={tl:.4f} | val_loss={vm['loss']:.4f} | "
                            f"val_acc={vm['acc']:.4f} | IoU_fg={vm['iou_fg']:.4f} | "
                            f"score={vm['score']:.4f} | "
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

        def _n_steps(self, n_samples: int) -> int:
            bs = max(1, self.cfg.batch_size)
            return max(1, (n_samples + bs - 1) // bs)

        def _batches(self, data_list: list, shuffle: bool):
            """Yield `Batch` objects of `batch_size` graphs each."""
            bs    = max(1, self.cfg.batch_size)
            order = (torch.randperm(len(data_list)).tolist() if shuffle
                     else list(range(len(data_list))))
            for i in range(0, len(order), bs):
                chunk = [data_list[j][0] for j in order[i:i + bs]]
                yield Batch.from_data_list(chunk).to(self.device)

        def _loss(self, batch, logits: torch.Tensor) -> torch.Tensor:
            """Evaluate the criterion, passing the extra supervision it accepts."""
            labels = batch.y
            if isinstance(self.criterion, TrimapLoss):
                return self.criterion(
                    logits, labels,
                    area=getattr(batch, "node_area", None),
                    fg_ratio=getattr(batch, "fg_ratio", None),
                    batch=getattr(batch, "batch", None),
                )
            return self.criterion(logits, labels)

        def _train_epoch(self, data_list: list) -> float:
            self.model.train()
            total_loss = 0.0
            n_batches  = 0

            for batch in self._batches(data_list, shuffle=True):
                self.optimizer.zero_grad(set_to_none=True)

                if self.scaler:
                    with autocast("cuda"):
                        loss = self._loss(batch, self.model(batch))
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss = self._loss(batch, self.model(batch))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.cfg.grad_clip)
                    self.optimizer.step()

                total_loss += float(loss.item())
                n_batches  += 1

                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()

            if self.scheduler and not isinstance(self.scheduler, (OneCycleLR, ReduceLROnPlateau)):
                self.scheduler.step()

            return total_loss / max(n_batches, 1)

        @torch.no_grad()
        def _eval_epoch(self, data_list: list) -> dict:
            self.model.eval()
            total_loss = 0.0
            n_batches  = 0
            all_preds, all_labels = [], []

            for batch in self._batches(data_list, shuffle=False):
                logits = self.model(batch)
                total_loss += float(self._loss(batch, logits).item())
                n_batches  += 1
                all_preds.append(logits.argmax(dim=-1).cpu())
                all_labels.append(batch.y.cpu())

            preds = torch.cat(all_preds)
            gts   = torch.cat(all_labels)
            acc   = (preds == gts).float().mean().item()
            ious  = _per_class_iou(preds.numpy(), gts.numpy(), n_classes=3)
            loss  = total_loss / max(n_batches, 1)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(loss)

            return {
                "loss":    loss,
                "acc":     acc,
                "iou_bg":  ious[CLASS_BG],
                "iou_unk": ious[CLASS_UNK],
                "iou_fg":  ious[CLASS_FG],
                # Selection criterion: the mean of the two decided classes.
                # Validation loss is a poor proxy here because the UNKNOWN
                # class dominates it while GrabCut resolves it downstream.
                "score":   0.5 * (ious[CLASS_FG] + ious[CLASS_BG]),
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

        def _save(self, filename: str, epoch: int, val_loss: Optional[float],
                  score: Optional[float] = None) -> None:
            path  = self.save_dir / filename
            state = {
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch":     epoch,
                "val_loss":  val_loss,
                "score":     score,
                "config":    asdict(self.cfg),
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
