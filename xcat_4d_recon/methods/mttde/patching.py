"""
Constrained k-means patching for the measure-theoretic training objective.

Adapted from: Measure-Theoretic-Time-Delay-Embedding-main/generate_patches_sparse.py

Partitions the delay-embedded training data into N_patches equal-size clusters.
Each patch is a mini-distribution (set of points) used to compute a Wasserstein
distance between network output and target PCA coefficients.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def build_patches(
    delay_coords: np.ndarray,
    pca_coefficients: np.ndarray,
    n_patches: int = 20,
    random_state: int = 0,
    max_iter: int = 100,
    patches_file: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cluster delay coordinates into equal-size patches.

    Parameters
    ----------
    delay_coords:
        Shape (N, embedding_dim).  Input side of the reconstruction map.
    pca_coefficients:
        Shape (N, n_pca).  Target side (ground-truth PCA coefficients).
    n_patches:
        Number of clusters.  Must evenly divide N.
    random_state:
        Seed for KMeansConstrained.
    max_iter:
        Max iterations for constrained k-means.
    patches_file:
        If provided, cache result at this path.

    Returns
    -------
    patch_inputs  : shape (n_patches, patch_size, embedding_dim) – torch.float32
    patch_outputs : shape (n_patches, patch_size, n_pca)         – torch.float32
    """
    # Load from cache
    if patches_file and Path(patches_file).exists():
        with open(patches_file, "rb") as f:
            patch_outputs, patch_inputs = pickle.load(f)
        print(f"Loaded patches from {patches_file}  "
              f"(shape inputs={tuple(patch_inputs.shape)}, outputs={tuple(patch_outputs.shape)})")
        return patch_inputs, patch_outputs

    try:
        from k_means_constrained import KMeansConstrained
    except ImportError as exc:
        raise ImportError(
            "k-means-constrained is required for patch generation.  "
            "Install with: pip install k-means-constrained"
        ) from exc

    N = delay_coords.shape[0]
    if N % n_patches != 0:
        # Trim to largest multiple of n_patches
        N_trimmed = (N // n_patches) * n_patches
        print(f"[patching] Trimming {N} → {N_trimmed} samples to divide evenly into {n_patches} patches.")
        delay_coords = delay_coords[:N_trimmed]
        pca_coefficients = pca_coefficients[:N_trimmed]
        N = N_trimmed

    patch_size = N // n_patches
    print(f"Building {n_patches} patches of {patch_size} samples each ...")

    clf = KMeansConstrained(
        n_clusters=n_patches,
        size_min=patch_size,
        size_max=patch_size,
        random_state=random_state,
        max_iter=max_iter,
    )
    point_idxs = clf.fit_predict(delay_coords)

    # Collect bin indices
    bins: list[list[int]] = [[] for _ in range(n_patches)]
    for i, cluster_id in enumerate(point_idxs):
        bins[cluster_id].append(i)

    patch_inputs_np = np.zeros((n_patches, patch_size, delay_coords.shape[1]), dtype=np.float32)
    patch_outputs_np = np.zeros((n_patches, patch_size, pca_coefficients.shape[1]), dtype=np.float32)
    for k in range(n_patches):
        patch_inputs_np[k] = delay_coords[bins[k]]
        patch_outputs_np[k] = pca_coefficients[bins[k]]

    patch_inputs = torch.tensor(patch_inputs_np, dtype=torch.float32)
    patch_outputs = torch.tensor(patch_outputs_np, dtype=torch.float32)

    if patches_file:
        Path(patches_file).parent.mkdir(parents=True, exist_ok=True)
        with open(patches_file, "wb") as f:
            pickle.dump([patch_outputs, patch_inputs], f)
        print(f"Saved patches → {patches_file}")

    return patch_inputs, patch_outputs
