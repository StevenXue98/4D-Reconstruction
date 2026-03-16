"""
PyTorch Dataset for POD-DeepONet training on 4D reconstruction data.

Each sample corresponds to one training timepoint and provides:
  - branch_input  : (n_delay, H, W) float32 tensor — n_delay downsampled
                    projection images stacked as channels, at delays
                    [t, t-τ, t-2τ, ..., t-(n-1)τ].  H × W is the training
                    projection resolution (e.g. 15×15).
  - target_coeffs : (n_pca,) float32 tensor — PCA coefficients of the
                    ground-truth volume at timepoint t.

All projection images are loaded at dataset initialisation (they are small
.npy files) and kept in memory.  PCA coefficients are provided as a
pre-computed array (computed once by fit_pca.py / run_deeponet.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class ProjectionDataset(Dataset):
    """Dataset mapping timepoint → (projection_stack, pca_coefficients).

    Parameters
    ----------
    timepoint_indices:
        Timepoints in this split.  Must satisfy ``t >= (n_delay-1)*tau`` for all t.
    proj_small_dir:
        Directory containing ``proj_small_<vol:03d>_angle_<a:02d>.npy`` files.
    angle_idx:
        Camera angle index to use.
    pca_coefficients:
        Pre-computed PCA coefficients, shape (n_total_timepoints, n_pca).
    tau:
        Takens time delay (samples).
    n_delay:
        Takens embedding dimension — number of delay images stacked.
    """

    def __init__(
        self,
        timepoint_indices: list[int],
        proj_small_dir: str,
        angle_idx: int,
        pca_coefficients: np.ndarray,
        tau: int,
        n_delay: int,
    ) -> None:
        self.indices = timepoint_indices
        self.pca_coefficients = pca_coefficients.astype(np.float32)

        proj_small_dir = Path(proj_small_dir)

        # ── Load all projection stacks upfront ────────────────────────────────
        # Each stack is (n_delay, H, W) — the H×W spatial image at each delay step.
        # Stored in a dict for O(1) access by timepoint index.
        self.branch_inputs: dict[int, np.ndarray] = {}
        proj_shape: tuple[int, int] | None = None

        for t in tqdm(timepoint_indices, desc="Loading projection stacks", leave=False):
            stack = []
            for j in range(n_delay):
                vol_idx = t - j * tau
                path = proj_small_dir / f"proj_small_{vol_idx:03d}_angle_{angle_idx:02d}.npy"
                img = np.load(str(path)).astype(np.float32)
                if proj_shape is None:
                    proj_shape = img.shape
                stack.append(img)
            # Shape: (n_delay, H, W)
            self.branch_inputs[t] = np.stack(stack, axis=0)

        h, w = proj_shape or (0, 0)
        print(
            f"ProjectionDataset: {len(timepoint_indices)} timepoints  "
            f"branch shape=({n_delay}, {h}, {w})  n_pca={pca_coefficients.shape[1]}"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.indices[idx]
        branch = torch.tensor(self.branch_inputs[t], dtype=torch.float32)
        target = torch.tensor(self.pca_coefficients[t], dtype=torch.float32)
        return branch, target
