"""
POD-DeepONet for 4D CT volume reconstruction.

In the Proper Orthogonal Decomposition (POD) variant of DeepONet:
  - The trunk basis is fixed to the PCA (POD) modes from the data.
  - The branch network learns to predict PCA coefficients from input data.
  - Reconstruction: volume = coefficients @ pca_components + pca_mean

This is equivalent to standard DeepONet but with a physics-informed trunk
derived from SVD/PCA of the training data, dramatically reducing the output
dimensionality from 11M voxels to n_pca coefficients.

Reference: Lu et al. (2022) "A comprehensive and fair comparison of two neural
operators (with practical extensions) based on FAIR benchmarks."
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class BranchNet(nn.Module):
    """Branch network encoding the input function to PCA coefficient space.

    Parameters
    ----------
    input_dim:
        Dimensionality of the branch input (e.g., embedding_dim for delay_vector,
        or slab_subsample_m for slab_subsample).
    hidden_dims:
        List of hidden layer widths.
    output_dim:
        Number of PCA components (= n_pca).
    activation:
        ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        act_cls = nn.ReLU if activation == "relu" else nn.Tanh
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act_cls()]
        for i in range(len(hidden_dims) - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act_cls()]
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PODDeepONet(nn.Module):
    """POD-DeepONet with fixed PCA trunk for high-dimensional volume reconstruction.

    The model predicts the PCA coefficient vector for a query timepoint given
    the branch input (delay embedding or CT slab).  The full volume is then
    reconstructed as a linear combination of PCA modes.

    Parameters
    ----------
    branch_input_dim:
        Size of branch network input.
    hidden_dims:
        Hidden layer widths for the branch network.
    n_pca:
        Number of PCA components.
    pca_components:
        Shape (n_pca, n_voxels).  Fixed trunk basis (not trained).
    pca_mean:
        Shape (n_voxels,).  Per-voxel mean.
    activation:
        Activation for branch net.
    """

    def __init__(
        self,
        branch_input_dim: int,
        hidden_dims: list[int],
        n_pca: int,
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.n_pca = n_pca
        self.branch = BranchNet(branch_input_dim, hidden_dims, n_pca, activation)

        if pca_components is not None:
            self.register_buffer(
                "pca_components",
                torch.tensor(pca_components, dtype=torch.float32),
            )  # (n_pca, n_voxels)
        else:
            self.pca_components = None

        if pca_mean is not None:
            self.register_buffer(
                "pca_mean",
                torch.tensor(pca_mean, dtype=torch.float32),
            )  # (n_voxels,)
        else:
            self.pca_mean = None

    def forward_coefficients(self, branch_input: torch.Tensor) -> torch.Tensor:
        """Predict PCA coefficients only (efficient for training with MSE loss).

        Parameters
        ----------
        branch_input: shape (batch, branch_input_dim)

        Returns: shape (batch, n_pca)
        """
        return self.branch(branch_input)

    def forward(
        self,
        branch_input: torch.Tensor,
        return_volume: bool = False,
    ) -> torch.Tensor:
        """Predict PCA coefficients, and optionally reconstruct full volume.

        Parameters
        ----------
        branch_input: shape (batch, branch_input_dim)
        return_volume: if True, reconstruct full volume (expensive).

        Returns
        -------
        If return_volume=False: coefficients, shape (batch, n_pca)
        If return_volume=True:  volumes, shape (batch, n_voxels)
        """
        coeffs = self.branch(branch_input)  # (batch, n_pca)
        if not return_volume:
            return coeffs

        if self.pca_components is None:
            raise RuntimeError("pca_components must be provided to reconstruct volumes.")
        # Linear combination: (batch, n_pca) @ (n_pca, n_voxels) + (n_voxels,)
        volumes = coeffs @ self.pca_components  # (batch, n_voxels)
        if self.pca_mean is not None:
            volumes = volumes + self.pca_mean.unsqueeze(0)
        return volumes

    def reconstruct_volume_np(
        self,
        coefficients: np.ndarray,
        volume_shape: tuple[int, ...],
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reconstruct a 3D volume from PCA coefficients (numpy, for inference).

        Parameters
        ----------
        coefficients: shape (n_pca,)
        volume_shape: e.g. (355, 280, 115)
        pca_components: optional override
        pca_mean: optional override

        Returns: shape volume_shape, float32
        """
        comps = pca_components if pca_components is not None else self.pca_components.cpu().numpy()
        mean = pca_mean if pca_mean is not None else (
            self.pca_mean.cpu().numpy() if self.pca_mean is not None else np.zeros(comps.shape[1])
        )
        flat = coefficients @ comps + mean  # (n_voxels,)
        return flat.reshape(volume_shape).astype(np.float32)
