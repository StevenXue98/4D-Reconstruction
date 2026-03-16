"""
Training loop for the Measure-Theoretic Time-Delay Embedding method.

Uses geomloss SamplesLoss("energy") — a Wasserstein-type energy distance —
computed patch-wise to match the distribution of reconstructed PCA coefficients
to the distribution of ground-truth PCA coefficients within each patch.

Adapted from: Measure-Theoretic-Time-Delay-Embedding-main/train_measures.py
"""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .network import ReconstructionNet


def train_mttde(
    patch_inputs: torch.Tensor,
    patch_outputs: torch.Tensor,
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 100,
    n_hidden_layers: int = 4,
    loss_type: str = "energy",
    n_iterations: int = 50000,
    learning_rate: float = 1e-3,
    log_every: int = 500,
    checkpoint_every: int = 5000,
    checkpoint_dir: Optional[str] = None,
    seed: int = 42,
    device: Optional[str] = None,
) -> tuple[ReconstructionNet, list[float]]:
    """Train the MTTDE reconstruction network.

    Parameters
    ----------
    patch_inputs:
        Shape (n_patches, patch_size, embedding_dim) – delay-coordinate patches.
    patch_outputs:
        Shape (n_patches, patch_size, n_pca) – target PCA coefficient patches.
    input_dim, output_dim:
        Network I/O dimensions.
    loss_type:
        ``"energy"`` (energy distance / kernel MMD) or ``"sinkhorn"`` (entropic OT).
    n_iterations:
        Total gradient steps.
    checkpoint_dir:
        If set, save .pth checkpoints here.

    Returns
    -------
    net : trained ReconstructionNet
    loss_history : list of float
    """
    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise ImportError(
            "geomloss is required for MTTDE training.  "
            "Install with: pip install geomloss"
        ) from exc

    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MTTDE on {device}")

    net = ReconstructionNet(input_dim, output_dim, hidden_dim, n_hidden_layers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    loss_fn = SamplesLoss(loss=loss_type)

    patch_inputs = patch_inputs.to(device)
    patch_outputs = patch_outputs.to(device)

    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    loss_history: list[float] = []
    t0 = time()

    pbar = tqdm(range(n_iterations), desc="MTTDE")
    for step in pbar:
        net.train()
        optimizer.zero_grad()

        # Forward on all patches: (n_patches, patch_size, output_dim)
        predicted = net(patch_inputs)
        # Wasserstein loss between predicted and target distributions in each patch
        L = loss_fn(predicted, patch_outputs).mean()
        L.backward()
        optimizer.step()

        loss_val = float(L.item())
        loss_history.append(loss_val)
        pbar.set_description(f"MTTDE loss={loss_val:.6e}")

        if log_every > 0 and step % log_every == 0:
            elapsed = time() - t0
            print(f"  step {step:6d}/{n_iterations}  loss={loss_val:.6e}  elapsed={elapsed:.1f}s")

        if checkpoint_dir and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            ckpt_path = Path(checkpoint_dir) / f"mttde_step{step+1:06d}.pth"
            torch.save(
                {
                    "step": step + 1,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_val,
                    "input_dim": input_dim,
                    "output_dim": output_dim,
                    "hidden_dim": hidden_dim,
                    "n_hidden_layers": n_hidden_layers,
                },
                ckpt_path,
            )

    total_time = time() - t0
    print(f"Training complete in {total_time:.1f}s  final loss={loss_history[-1]:.6e}")

    # Save final model
    if checkpoint_dir:
        final_path = Path(checkpoint_dir) / "mttde_final.pth"
        torch.save(
            {
                "step": n_iterations,
                "model_state_dict": net.state_dict(),
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden_dim": hidden_dim,
                "n_hidden_layers": n_hidden_layers,
                "loss_history": loss_history,
                "training_time_s": total_time,
            },
            final_path,
        )
        print(f"Saved final model → {final_path}")

    return net.cpu(), loss_history


def load_mttde(checkpoint_path: str, device: str = "cpu") -> ReconstructionNet:
    """Load a trained ReconstructionNet from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    net = ReconstructionNet(
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        hidden_dim=ckpt["hidden_dim"],
        n_hidden_layers=ckpt["n_hidden_layers"],
    )
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net
