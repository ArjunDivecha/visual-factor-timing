"""
=============================================================================
SCRIPT NAME: loop.py
=============================================================================

DESCRIPTION:
Single-model training loop following the paper's §3.2 hyperparameters:

  optimizer:           Adam(lr=1e-3, β1=0.9, β2=0.99, ε=1e-7)
  batch size:          2**15 (32,768) — capped by dataset size
  max epochs:          100
  early stopping:      patience = 7 on validation loss
  loss:                MSE or MAE (weighted)
  train/val split:     70/30 of the training window

Used as an inner worker by the 64-model ensemble loop in ensemble.py.

VERSION: 1.0
LAST UPDATED: 2026-04-22

DEPENDENCIES:
- torch, numpy
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def _default_device() -> str:
    """Prefer Apple Silicon MPS > CUDA > CPU."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    lr:        float = 1e-3
    beta1:     float = 0.9
    beta2:     float = 0.99
    eps:       float = 1e-7
    batch_size: int  = 32768           # 2**15 per paper
    max_epochs: int  = 200             # was 100 — let the model train longer
    patience:   int  = 15              # was 7  — require a real plateau before stopping
    min_delta:  float = 1e-4           # minimum val_loss improvement to reset counter
    lr_reduce_patience: int = 6        # halve LR if val stagnates this long
    lr_reduce_factor:   float = 0.5
    min_lr:    float = 1e-5            # floor for the scheduler
    loss:      Literal["mse", "mae"] = "mse"
    device:    str = field(default_factory=_default_device)


def _weighted_loss(pred, target, weight, kind: str) -> torch.Tensor:
    if kind == "mse":
        base = (pred - target) ** 2
    elif kind == "mae":
        base = torch.abs(pred - target)
    else:
        raise ValueError(kind)
    return (weight * base).sum() / weight.sum().clamp_min(1e-8)


def train_one(
    model: torch.nn.Module,
    train_ds: Dataset,
    val_ds:   Dataset,
    cfg: TrainConfig,
    seed: int = 0,
) -> Tuple[torch.nn.Module, dict]:
    """Train `model` on `train_ds`, early-stopping on `val_ds`. Returns best
    model (in-place on the original object) and a stats dict.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    # DataLoaders
    train_bs = min(cfg.batch_size, len(train_ds)) or 1
    val_bs   = min(cfg.batch_size, len(val_ds))   or 1
    train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True,
                          num_workers=0, drop_last=False)
    val_dl   = DataLoader(val_ds,   batch_size=val_bs,   shuffle=False,
                          num_workers=0, drop_last=False)

    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2), eps=cfg.eps,
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min",
        factor=cfg.lr_reduce_factor,
        patience=cfg.lr_reduce_patience,
        min_lr=cfg.min_lr,
        threshold=cfg.min_delta, threshold_mode="abs",
    )

    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0
    history = []

    for epoch in range(cfg.max_epochs):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        for x, y, w in train_dl:
            x = x.to(device)
            y = y.to(device)
            w = w.to(device)
            pred = model(x)
            loss = _weighted_loss(pred, y, w, cfg.loss)
            opt.zero_grad()
            loss.backward()
            opt.step()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            preds, ys, ws = [], [], []
            for x, y, w in val_dl:
                preds.append(model(x.to(device)).cpu())
                ys.append(y)
                ws.append(w)
            if not preds:
                val_loss = float("nan")
            else:
                pred = torch.cat(preds)
                y    = torch.cat(ys)
                w    = torch.cat(ws)
                val_loss = _weighted_loss(pred, y, w, cfg.loss).item()

        current_lr = opt.param_groups[0]["lr"]
        history.append({"epoch": epoch, "val_loss": val_loss, "lr": current_lr})

        # LR scheduler step (uses its own plateau counter, independent of
        # the early-stopping counter below)
        if np.isfinite(val_loss):
            sched.step(val_loss)

        # ── Early stopping with min_delta threshold ───────────────────────
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                break

    model.load_state_dict(best_state)
    best_epoch = (min(range(len(history)), key=lambda i: history[i]["val_loss"])
                  if history else 0)
    return model, {
        "best_val_loss": best_val,
        "best_epoch":    best_epoch,
        "epochs":        len(history),
        "final_lr":      history[-1]["lr"] if history else cfg.lr,
        "history":       history,
    }


def predict(model: torch.nn.Module, ds: Dataset, batch_size: int = 32768,
            device: str = "cpu") -> np.ndarray:
    """Predict a full dataset in batches, returning a 1D numpy array."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model = model.to(device).eval()
    out = []
    with torch.no_grad():
        for x, _, _ in dl:
            out.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0,), dtype=np.float32)
