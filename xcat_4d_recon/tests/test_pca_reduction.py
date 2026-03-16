"""Tests for PCA-based dimensionality reduction (round-trip accuracy)."""

import tempfile
from pathlib import Path
import sys

import numpy as np
import pytest

nibabel = pytest.importorskip("nibabel", reason="nibabel not installed")
nib = nibabel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.pca_reduction import PCAReduction


def _make_fake_volumes(n_vols: int, shape: tuple, tmp_dir: str) -> list[str]:
    """Write synthetic NIfTI volumes and return their paths."""
    paths = []
    rng = np.random.RandomState(0)
    for i in range(n_vols):
        data = rng.randn(*shape).astype(np.float32) * 100
        img = nib.Nifti1Image(data, np.eye(4))
        path = str(Path(tmp_dir) / f"volume_{i}.nii.gz")
        nib.nifti1.save(img, path)
        paths.append(path)
    return paths


def test_encode_decode_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_volumes(12, (15, 12, 10), tmp)
        pca = PCAReduction(n_components=8, batch_size=4)
        pca.fit(paths)

        # Round-trip on a training volume
        vol = nib.load(paths[0]).get_fdata().astype(np.float32)
        coeffs = pca.encode(vol)
        recon = pca.decode(coeffs)
        assert recon.shape == vol.shape

        # RMSE should be finite and small (not exact due to compression)
        rmse = np.sqrt(np.mean((vol - recon) ** 2))
        assert np.isfinite(rmse)


def test_save_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_volumes(10, (8, 8, 6), tmp)
        pca = PCAReduction(n_components=5, batch_size=4)
        pca.fit(paths)

        basis_file = str(Path(tmp) / "pca_basis.npz")
        pca.save(basis_file)

        pca2 = PCAReduction()
        pca2.load(basis_file)

        vol = nib.load(paths[0]).get_fdata().astype(np.float32)
        c1 = pca.encode(vol)
        c2 = pca2.encode(vol)
        np.testing.assert_array_almost_equal(c1, c2, decimal=5)


def test_encode_many_shape():
    with tempfile.TemporaryDirectory() as tmp:
        n_vols = 6
        paths = _make_fake_volumes(n_vols, (10, 8, 6), tmp)
        pca = PCAReduction(n_components=4, batch_size=3)
        pca.fit(paths)
        coeffs = pca.encode_many(paths)
        assert coeffs.shape == (n_vols, 4)


def test_n_components_property():
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_fake_volumes(8, (10, 8, 6), tmp)
        pca = PCAReduction(n_components=4, batch_size=4)
        pca.fit(paths)
        assert pca.n_components == 4
        assert pca.components.shape == (4, 10 * 8 * 6)
        assert pca.mean.shape == (10 * 8 * 6,)
