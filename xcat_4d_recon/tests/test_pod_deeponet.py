"""Tests for POD-DeepONet architecture."""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.deeponet.pod_deeponet import PODDeepONet, BranchNet


def test_branch_net_output_shape():
    net = BranchNet(input_dim=8, hidden_dims=[32, 32], output_dim=16)
    x = torch.randn(4, 8)
    out = net(x)
    assert out.shape == (4, 16)


def test_pod_deeponet_coeff_shape():
    model = PODDeepONet(branch_input_dim=6, hidden_dims=[32, 16], n_pca=10)
    x = torch.randn(3, 6)
    coeffs = model.forward_coefficients(x)
    assert coeffs.shape == (3, 10)


def test_pod_deeponet_volume_reconstruction():
    n_pca, n_voxels = 5, 100
    comps = np.random.randn(n_pca, n_voxels).astype(np.float32)
    mean = np.random.randn(n_voxels).astype(np.float32)

    model = PODDeepONet(
        branch_input_dim=4,
        hidden_dims=[16],
        n_pca=n_pca,
        pca_components=comps,
        pca_mean=mean,
    )
    x = torch.randn(2, 4)
    volumes = model(x, return_volume=True)
    assert volumes.shape == (2, n_voxels)


def test_pod_deeponet_reconstruct_np():
    n_pca, n_voxels = 4, 60
    comps = np.random.randn(n_pca, n_voxels).astype(np.float32)
    mean = np.random.randn(n_voxels).astype(np.float32)
    model = PODDeepONet(branch_input_dim=4, hidden_dims=[16], n_pca=n_pca,
                        pca_components=comps, pca_mean=mean)
    coeffs = np.random.randn(n_pca).astype(np.float32)
    vol = model.reconstruct_volume_np(coeffs, volume_shape=(3, 4, 5))
    assert vol.shape == (3, 4, 5)


def test_mttde_network_output_shape():
    from methods.mttde.network import ReconstructionNet
    net = ReconstructionNet(input_dim=5, output_dim=32, hidden_dim=100, n_hidden_layers=4)
    x = torch.randn(10, 5)
    out = net(x)
    assert out.shape == (10, 32)
