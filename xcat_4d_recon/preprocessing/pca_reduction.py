"""
PCA-based dimensionality reduction for 3D CT volumes.

Uses IncrementalPCA to handle the high-dimensional (11M+ voxel) volumes without
requiring all data in memory simultaneously.  Fitted on the training split only;
shared by both MTTDE and DeepONet downstream methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm


class PCAReduction:
    """Incremental PCA wrapper for volumetric CT data.

    Parameters
    ----------
    n_components:
        Number of PCA components to retain.
    batch_size:
        Number of volumes to process per IncrementalPCA partial_fit call.
    """

    def __init__(self, n_components: int = 64, batch_size: int = 10) -> None:
        self.n_components = n_components
        self.batch_size = batch_size
        self._pca: Optional[IncrementalPCA] = None
        self._volume_shape: Optional[tuple[int, ...]] = None
        self._hu_min: Optional[float] = None
        self._hu_max: Optional[float] = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(
        self,
        volume_paths: list[str],
        normalisation_file: Optional[str] = None,
    ) -> "PCAReduction":
        """Fit IncrementalPCA on a list of NIfTI volume paths.

        Parameters
        ----------
        volume_paths:
            Ordered list of .nii.gz paths (training volumes only).
        normalisation_file:
            If provided, save HU min/max statistics here for later normalisation.
        """
        print(f"Fitting PCA (n_components={self.n_components}) on {len(volume_paths)} volumes ...")

        # First pass: determine volume shape and HU range
        sample = nib.load(volume_paths[0]).get_fdata().astype(np.float32)
        self._volume_shape = sample.shape
        n_voxels = int(np.prod(self._volume_shape))

        self._pca = IncrementalPCA(
            n_components=self.n_components,
            batch_size=max(self.n_components + 1, self.batch_size),
        )

        # Collect HU stats and partial-fit in batches
        hu_min, hu_max = np.inf, -np.inf
        for start in tqdm(range(0, len(volume_paths), self.batch_size), desc="PCA fit"):
            batch_paths = volume_paths[start : start + self.batch_size]
            batch = np.stack(
                [nib.load(p).get_fdata().astype(np.float32).ravel() for p in batch_paths],
                axis=0,
            )  # (batch, n_voxels)
            hu_min = min(hu_min, float(batch.min()))
            hu_max = max(hu_max, float(batch.max()))
            self._pca.partial_fit(batch)

        self._hu_min = hu_min
        self._hu_max = hu_max

        if normalisation_file:
            Path(normalisation_file).parent.mkdir(parents=True, exist_ok=True)
            with open(normalisation_file, "w") as f:
                json.dump({"hu_min": hu_min, "hu_max": hu_max}, f, indent=2)
            print(f"Saved normalisation stats → {normalisation_file}")

        print(
            f"PCA fitted.  Explained variance (first {self.n_components} PCs): "
            f"{self._pca.explained_variance_ratio_.sum():.3%}"
        )
        return self

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, volume: np.ndarray) -> np.ndarray:
        """Project a 3D volume to PCA coefficients.

        Parameters
        ----------
        volume: shape (x, y, z) float32
        Returns: shape (n_components,) float64
        """
        assert self._pca is not None, "Call fit() or load() first."
        return self._pca.transform(volume.ravel()[np.newaxis, :])[0]

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct a 3D volume from PCA coefficients.

        Parameters
        ----------
        coefficients: shape (n_components,)
        Returns: shape (*volume_shape) float32
        """
        assert self._pca is not None, "Call fit() or load() first."
        flat = self._pca.inverse_transform(coefficients[np.newaxis, :])[0]
        return flat.reshape(self._volume_shape).astype(np.float32)

    def encode_many(self, volume_paths: list[str]) -> np.ndarray:
        """Encode a list of NIfTI paths into a coefficient matrix.

        Returns: shape (n_volumes, n_components)
        """
        coefficients = []
        for p in tqdm(volume_paths, desc="Encoding"):
            vol = nib.load(p).get_fdata().astype(np.float32)
            coefficients.append(self.encode(vol))
        return np.stack(coefficients, axis=0)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, basis_file: str) -> None:
        """Save PCA components, mean, and metadata to an .npz file."""
        assert self._pca is not None
        Path(basis_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            basis_file,
            components=self._pca.components_,           # (n_components, n_voxels)
            mean=self._pca.mean_,                        # (n_voxels,)
            explained_variance=self._pca.explained_variance_,
            explained_variance_ratio=self._pca.explained_variance_ratio_,
            volume_shape=np.array(self._volume_shape),
            hu_min=np.array(self._hu_min),
            hu_max=np.array(self._hu_max),
        )
        print(f"Saved PCA basis → {basis_file}")

    def load(self, basis_file: str) -> "PCAReduction":
        """Restore a previously saved PCA basis."""
        data = np.load(basis_file)
        self._volume_shape = tuple(data["volume_shape"].tolist())
        n_voxels = int(np.prod(self._volume_shape))

        self._pca = IncrementalPCA(n_components=data["components"].shape[0])
        # Restore internal state expected by sklearn's IncrementalPCA
        self._pca.components_ = data["components"]
        self._pca.mean_ = data["mean"]
        self._pca.explained_variance_ = data["explained_variance"]
        self._pca.explained_variance_ratio_ = data["explained_variance_ratio"]
        self._pca.n_components_ = data["components"].shape[0]
        self._pca.n_samples_seen_ = 0  # not needed for transform
        self._pca.n_features_in_ = n_voxels
        self._pca.singular_values_ = np.sqrt(
            data["explained_variance"] * (self._pca.n_samples_seen_ + 1)
        )
        self._hu_min = float(data["hu_min"])
        self._hu_max = float(data["hu_max"])
        self.n_components = data["components"].shape[0]
        print(f"Loaded PCA basis from {basis_file}  (n_components={self.n_components})")
        return self

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def components(self) -> np.ndarray:
        """PCA components array, shape (n_components, n_voxels)."""
        assert self._pca is not None
        return self._pca.components_

    @property
    def mean(self) -> np.ndarray:
        """Per-voxel mean, shape (n_voxels,)."""
        assert self._pca is not None
        return self._pca.mean_
