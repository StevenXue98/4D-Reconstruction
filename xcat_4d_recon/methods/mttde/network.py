"""
Neural network architecture for MTTDE reconstruction.

Adapted from: Measure-Theoretic-Time-Delay-Embedding-main/train_measures.py

Architecture: fully-connected network with tanh activations.
  Input:  embedding dimension (n)
  Hidden: n_hidden_layers × hidden_dim nodes, tanh
  Output: n_pca (PCA coefficient dimension)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReconstructionNet(nn.Module):
    """MLP that maps delay-coordinate vectors to PCA coefficients.

    Parameters
    ----------
    input_dim:
        Dimensionality of the time-delay embedding (tau * n).
    output_dim:
        Number of PCA components to predict.
    hidden_dim:
        Width of each hidden layer.
    n_hidden_layers:
        Number of hidden layers (all with tanh activation).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 100,
        n_hidden_layers: int = 4,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
