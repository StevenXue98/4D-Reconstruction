"""
POD-DeepONet with CNN branch for 4D volume reconstruction.

Architecture
------------
Branch network (CNN)
    Input  : (batch, n_delay, H, W)
             n_delay channels — one downsampled projection image per Takens delay step.
             H × W is the training resolution (e.g. 15×15), but the network is
             resolution-invariant: AdaptiveAvgPool2d collapses any spatial size to a
             fixed (pool_size × pool_size) feature map, so the same weights can process
             larger projection images at inference without retraining.
    Output : (batch, n_pca) — PCA coefficient vector

Trunk (fixed POD basis)
    The trunk is NOT a network — it is the fixed PCA basis computed from the
    training volumes (Proper Orthogonal Decomposition modes).  This is the
    POD-DeepONet variant: the trunk basis is data-derived rather than learned,
    which gives the best linear approximation in L2 for a given number of modes.

Reconstruction
    volume = coefficients @ pca_components + pca_mean

Resolution invariance
    The branch CNN processes any (H, W) input because AdaptiveAvgPool2d(pool_size)
    always outputs a pool_size × pool_size spatial map regardless of input size.
    Train on n×n downsampled projections; at inference pass full-resolution images
    without any architectural change.

Reference: Lu et al. (2022), "A comprehensive and fair comparison of two neural
operators (with practical extensions) based on FAIR benchmarks."
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class BranchCNN(nn.Module):
    """CNN branch: (batch, n_delay, H, W) → (batch, n_pca).

    Processes n_delay projection images as channels of a 2D image.
    Three conv layers with ReLU, followed by AdaptiveAvgPool2d and a linear head.
    The adaptive pooling layer is what confers input-resolution invariance.

    Parameters
    ----------
    n_delay:
        Number of delay images stacked as input channels (Takens embedding dim).
    n_pca:
        Output dimensionality (number of PCA / POD components).
    base_channels:
        Number of feature maps in the first conv layer; doubled at each stage.
    pool_size:
        Spatial size of the adaptive pooling output.  Controls the capacity of
        the feature vector fed to the linear head:
        feature_dim = base_channels * 4 * pool_size * pool_size.
    """

    def __init__(
        self,
        n_delay: int,
        n_pca: int,
        base_channels: int = 32,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.conv = nn.Sequential(
            nn.Conv2d(n_delay, c1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),      nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),      nn.ReLU(),
        )
        # Collapses any (H, W) → (pool_size, pool_size): resolution invariant
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.head = nn.Linear(c3 * pool_size * pool_size, n_pca)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_delay, H, W) → (batch, n_pca)"""
        x = self.conv(x)          # (batch, c3, H, W)
        x = self.pool(x)          # (batch, c3, pool_size, pool_size)
        x = x.flatten(1)          # (batch, c3 * pool_size * pool_size)
        return self.head(x)       # (batch, n_pca)


class PODDeepONet(nn.Module):
    """POD-DeepONet: CNN branch over projection stacks, fixed POD trunk.

    Parameters
    ----------
    n_delay:
        Takens embedding dimension (number of delay images).
    n_pca:
        Number of PCA / POD modes.
    pca_components:
        Shape (n_pca, n_voxels).  Fixed trunk basis (not trained).
    pca_mean:
        Shape (n_voxels,).  Per-voxel mean.
    base_channels:
        Conv feature maps at first stage (doubles per stage).
    pool_size:
        Adaptive pooling output size.
    """

    def __init__(
        self,
        n_delay: int,
        n_pca: int,
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
        base_channels: int = 32,
        pool_size: int = 4,
    ) -> None:
        super().__init__()
        self.n_pca = n_pca
        self.branch = BranchCNN(n_delay, n_pca, base_channels, pool_size)

        if pca_components is not None:
            self.register_buffer(
                "pca_components",
                torch.tensor(pca_components, dtype=torch.float32),
            )
        else:
            self.pca_components = None

        if pca_mean is not None:
            self.register_buffer(
                "pca_mean",
                torch.tensor(pca_mean, dtype=torch.float32),
            )
        else:
            self.pca_mean = None

    def forward_coefficients(self, branch_input: torch.Tensor) -> torch.Tensor:
        """Predict PCA coefficients.

        Parameters
        ----------
        branch_input: (batch, n_delay, H, W)

        Returns: (batch, n_pca)
        """
        return self.branch(branch_input)

    def forward(
        self,
        branch_input: torch.Tensor,
        return_volume: bool = False,
    ) -> torch.Tensor:
        """Predict PCA coefficients, optionally reconstruct full volume.

        Parameters
        ----------
        branch_input: (batch, n_delay, H, W)
        return_volume: if True, reconstruct full volume (expensive for large volumes).

        Returns
        -------
        If return_volume=False: coefficients (batch, n_pca)
        If return_volume=True:  volumes (batch, n_voxels)
        """
        coeffs = self.branch(branch_input)
        if not return_volume:
            return coeffs
        if self.pca_components is None:
            raise RuntimeError("pca_components must be provided to reconstruct volumes.")
        volumes = coeffs @ self.pca_components
        if self.pca_mean is not None:
            volumes = volumes + self.pca_mean.unsqueeze(0)
        return volumes
