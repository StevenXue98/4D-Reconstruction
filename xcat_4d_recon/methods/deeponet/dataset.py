"""
PyTorch Dataset for POD-DeepONet training on 4D XCAT data.

Each sample corresponds to one training timepoint and provides:
  - branch_input : 1-D vector encoding the acquired data at that timepoint
                   (either a delay-coordinate vector or a sub-sampled CT slab).
  - target_coeffs: PCA coefficients of the ground-truth volume.

The trunk (POD basis) is fixed and not part of per-sample data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class CT4DDataset(Dataset):
    """Dataset mapping timepoint → (branch_input, pca_coefficients).

    Parameters
    ----------
    timepoint_indices:
        List of timepoint indices to include (e.g., training split).
    pca_coefficients:
        Pre-computed PCA coefficients, shape (n_total_timepoints, n_pca).
    branch_method:
        ``"delay_vector"`` – use pre-computed delay embedding vectors.
        ``"slab_subsample"`` – load and subsample CT slabs.
    delay_matrix:
        Shape (n_aligned, n_embedding).  Required for ``"delay_vector"``.
        Row i corresponds to timepoint (n-1)*tau + i in the original signal.
    delay_offset:
        Number of samples trimmed at the start of the surrogate signal when
        building the delay matrix (= (n-1)*tau).  Used to align delay rows with
        timepoint indices.
    slab_dir:
        Directory containing slab_*.nii.gz (required for ``"slab_subsample"``).
    slab_subsample_m:
        Number of voxels to randomly sub-sample from each slab.
    subsample_seed:
        RNG seed for reproducible sub-sampling.
    """

    def __init__(
        self,
        timepoint_indices: list[int],
        pca_coefficients: np.ndarray,
        branch_method: str = "delay_vector",
        delay_matrix: Optional[np.ndarray] = None,
        delay_offset: int = 0,
        slab_dir: Optional[str] = None,
        slab_subsample_m: int = 512,
        subsample_seed: int = 42,
    ) -> None:
        self.indices = timepoint_indices
        self.pca_coefficients = pca_coefficients.astype(np.float32)
        self.branch_method = branch_method
        self.delay_matrix = delay_matrix
        self.delay_offset = delay_offset
        self.slab_dir = Path(slab_dir) if slab_dir else None
        self.slab_subsample_m = slab_subsample_m

        if branch_method == "delay_vector":
            if delay_matrix is None:
                raise ValueError("delay_matrix is required for branch_method='delay_vector'.")

        elif branch_method == "slab_subsample":
            if slab_dir is None:
                raise ValueError("slab_dir is required for branch_method='slab_subsample'.")
            # Pre-compute fixed random sub-sample indices (consistent across timepoints)
            rng = np.random.RandomState(subsample_seed)
            # We don't know slab shape yet; sub-sampling is done lazily
            self._rng = rng
            self._slab_idx_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]

        # ── Branch input ──────────────────────────────────────────────────────
        if self.branch_method == "delay_vector":
            row = t - self.delay_offset
            if row < 0 or row >= len(self.delay_matrix):
                raise IndexError(
                    f"Timepoint {t} maps to delay_matrix row {row}, which is out of bounds "
                    f"(delay_matrix has {len(self.delay_matrix)} rows).  "
                    "Check that delay_offset = (n-1)*tau matches the trimmed delay matrix."
                )
            branch = torch.tensor(self.delay_matrix[row], dtype=torch.float32)

        elif self.branch_method == "slab_subsample":
            slab_path = self.slab_dir / f"slab_{t}.nii.gz"
            slab = nib.load(str(slab_path)).get_fdata().astype(np.float32).ravel()
            n_voxels = len(slab)
            if t not in self._slab_idx_cache:
                self._slab_idx_cache[t] = self._rng.choice(
                    n_voxels, size=min(self.slab_subsample_m, n_voxels), replace=False
                )
            branch = torch.tensor(slab[self._slab_idx_cache[t]], dtype=torch.float32)

        else:
            raise ValueError(f"Unknown branch_method: {self.branch_method!r}")

        # ── Target PCA coefficients ───────────────────────────────────────────
        target = torch.tensor(self.pca_coefficients[t], dtype=torch.float32)

        return branch, target
