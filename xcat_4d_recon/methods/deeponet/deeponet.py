"""
DeepONet for 4D volume reconstruction from 2D projection image stacks.

Architecture
------------
Branch network
    Input  : flattened spatial-temporal projection stack
             shape (H_small * W_small * n_delay,)
             where H_small × W_small is the downsampled projection image size
             and n_delay is the Takens embedding dimension.
    Output : n_basis coefficient vector

Trunk network
    Input  : normalised (x, y, z) query coordinates in [-1, 1]^3
    Output : n_basis learned basis function values at that point

Prediction at a query point q given branch input u:
    out(q) = dot(branch(u), trunk(q)) + bias

This is resolution-invariant: the trunk can be queried at any spatial
location, so the model can reconstruct volumes at any resolution without
retraining.  During training, a random subset of voxels is sampled from
the ground-truth volume for efficiency.

Reference: Lu et al. (2019) "DeepONet: Learning nonlinear operators via
DeepONet based on the universal approximation theorem of operators."
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: str) -> nn.Sequential:
    act_cls = nn.ReLU if activation == "relu" else nn.Tanh
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dims[0]), act_cls()]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), act_cls()]
    layers.append(nn.Linear(hidden_dims[-1], output_dim))
    return nn.Sequential(*layers)


class BranchNet(nn.Module):
    """Maps a flattened spatial-temporal projection stack to n_basis coefficients.

    Parameters
    ----------
    input_dim:
        H_small * W_small * n_delay
    hidden_dims:
        Hidden layer widths.
    n_basis:
        Number of basis functions (output dimension).
    activation:
        ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_basis: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.net = _mlp(input_dim, hidden_dims, n_basis, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) → (batch, n_basis)"""
        return self.net(x)


class TrunkNet(nn.Module):
    """Maps normalised (x, y, z) coordinates to n_basis basis function values.

    Parameters
    ----------
    hidden_dims:
        Hidden layer widths.
    n_basis:
        Number of basis functions (output dimension).
    activation:
        ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        hidden_dims: list[int],
        n_basis: int,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        # Input is always 3D (x, y, z)
        self.net = _mlp(3, hidden_dims, n_basis, activation)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (..., 3) → (..., n_basis)"""
        return self.net(coords)


class DeepONet(nn.Module):
    """Resolution-invariant DeepONet for 4D volume reconstruction.

    Parameters
    ----------
    branch_input_dim:
        H_small * W_small * n_delay.
    branch_hidden_dims:
        Hidden layer widths for branch net.
    trunk_hidden_dims:
        Hidden layer widths for trunk net.
    n_basis:
        Number of learned basis functions.
    activation:
        Activation for both networks.
    """

    def __init__(
        self,
        branch_input_dim: int,
        branch_hidden_dims: list[int],
        trunk_hidden_dims: list[int],
        n_basis: int = 64,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.branch = BranchNet(branch_input_dim, branch_hidden_dims, n_basis, activation)
        self.trunk = TrunkNet(trunk_hidden_dims, n_basis, activation)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_input: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Predict field values at query locations.

        Parameters
        ----------
        branch_input : (batch, branch_input_dim)
        query_coords : (batch, M, 3)  or  (M, 3) for single-sample inference

        Returns
        -------
        values : (batch, M)  or  (M,) to match query_coords shape
        """
        b = self.branch(branch_input)  # (batch, n_basis)

        if query_coords.dim() == 2:
            # Single-sample inference: (M, 3) → (M, n_basis)
            t = self.trunk(query_coords)             # (M, n_basis)
            return (t @ b.squeeze(0)) + self.bias    # (M,)

        # Batched: (batch, M, 3)
        t = self.trunk(query_coords)                 # (batch, M, n_basis)
        out = (t * b.unsqueeze(1)).sum(-1)           # (batch, M)
        return out + self.bias

    def predict_volume(
        self,
        branch_input: torch.Tensor,
        volume_shape: tuple[int, int, int],
        chunk_size: int = 65536,
        device: str = "cpu",
    ) -> np.ndarray:
        """Reconstruct a full 3D volume by querying all voxel coordinates.

        The trunk is evaluated in chunks to avoid OOM errors.  The branch
        is evaluated once; each chunk then does a single matrix-vector multiply
        against the cached branch output.

        Parameters
        ----------
        branch_input : (1, branch_input_dim) or (branch_input_dim,)
        volume_shape : (X, Y, Z)
        chunk_size   : number of voxels evaluated per trunk forward pass

        Returns
        -------
        volume : shape (X, Y, Z)  float32
        """
        self.eval()
        X, Y, Z = volume_shape

        if branch_input.dim() == 1:
            branch_input = branch_input.unsqueeze(0)
        branch_input = branch_input.to(device)

        with torch.no_grad():
            b = self.branch(branch_input).squeeze(0)  # (n_basis,)

        # Build normalised coordinate grid
        xs = torch.linspace(-1.0, 1.0, X)
        ys = torch.linspace(-1.0, 1.0, Y)
        zs = torch.linspace(-1.0, 1.0, Z)
        grid_x, grid_y, grid_z = torch.meshgrid(xs, ys, zs, indexing="ij")
        coords_flat = torch.stack(
            [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=1
        )  # (X*Y*Z, 3)

        values = torch.empty(X * Y * Z, dtype=torch.float32)
        for start in range(0, len(coords_flat), chunk_size):
            chunk = coords_flat[start : start + chunk_size].to(device)
            with torch.no_grad():
                t_chunk = self.trunk(chunk)                   # (Q, n_basis)
                v_chunk = (t_chunk @ b) + self.bias           # (Q,)
            values[start : start + chunk_size] = v_chunk.cpu()

        return values.numpy().reshape(volume_shape)
