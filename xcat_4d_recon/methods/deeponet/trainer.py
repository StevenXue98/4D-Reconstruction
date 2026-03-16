"""
Training loop for POD-DeepONet on 4D CT reconstruction.

Trains the branch network to minimise MSE between predicted and true PCA
coefficients.  Training in PCA coefficient space is equivalent to a
weighted MSE in voxel space (weighted by explained variance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .pod_deeponet import PODDeepONet


def train_deeponet(
    model: PODDeepONet,
    train_loader: DataLoader,
    n_epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    lr_scheduler: str = "cosine",
    log_every: int = 10,
    checkpoint_every: int = 50,
    checkpoint_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> tuple[PODDeepONet, list[float]]:
    """Train a PODDeepONet model.

    Parameters
    ----------
    model:
        Initialised PODDeepONet.
    train_loader:
        DataLoader yielding (branch_input, target_coefficients) batches.
    n_epochs:
        Number of full passes over the training data.
    lr_scheduler:
        ``"cosine"`` (cosine annealing), ``"step"`` (step decay), or ``"none"``.
    checkpoint_dir:
        If set, save .pth checkpoints here.

    Returns
    -------
    model : trained PODDeepONet (on CPU)
    epoch_losses : list of mean training loss per epoch
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training DeepONet on {device}")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=n_epochs // 5, gamma=0.5)
    else:
        scheduler = None

    loss_fn = nn.MSELoss()

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    epoch_losses: list[float] = []

    for epoch in tqdm(range(1, n_epochs + 1), desc="DeepONet"):
        model.train()
        batch_losses: list[float] = []

        for branch_input, target in train_loader:
            branch_input = branch_input.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model.forward_coefficients(branch_input)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(batch_losses))
        epoch_losses.append(epoch_loss)

        if scheduler is not None:
            scheduler.step()

        if log_every > 0 and epoch % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:4d}/{n_epochs}  loss={epoch_loss:.6e}  lr={lr:.2e}")

        if checkpoint_dir and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"deeponet_epoch{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                ckpt_path,
            )

    print(f"Training complete.  Final epoch loss={epoch_losses[-1]:.6e}")

    # Save final model
    if checkpoint_dir:
        final_path = Path(checkpoint_dir) / "deeponet_final.pth"
        torch.save(
            {
                "epoch": n_epochs,
                "model_state_dict": model.state_dict(),
                "epoch_losses": epoch_losses,
            },
            final_path,
        )
        print(f"Saved final model → {final_path}")

    return model.cpu(), epoch_losses


def load_deeponet(
    checkpoint_path: str,
    model: PODDeepONet,
    device: str = "cpu",
) -> PODDeepONet:
    """Load a saved PODDeepONet checkpoint into an existing model instance."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
