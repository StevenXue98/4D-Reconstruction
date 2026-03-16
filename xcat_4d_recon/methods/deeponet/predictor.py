"""
Inference for POD-DeepONet: projection stack → PCA coefficients → 3D volume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .pod_deeponet import PODDeepONet
from preprocessing.pca_reduction import PCAReduction


def predict_deeponet(
    model: PODDeepONet,
    pca: PCAReduction,
    branch_inputs: np.ndarray,
    test_indices: list[int],
    output_dir: str,
    ref_affine: Optional[np.ndarray] = None,
    batch_size: int = 8,
) -> None:
    """Run DeepONet inference and save reconstructed NIfTI volumes.

    Parameters
    ----------
    model:
        Trained PODDeepONet (on CPU).
    pca:
        Fitted PCAReduction instance (for decoding coefficients → volume).
    branch_inputs:
        Pre-computed branch inputs, shape (n_timepoints, n_delay, H, W).
        Indexed by absolute timepoint.  H × W may differ from training
        resolution — the CNN branch handles any spatial size.
    test_indices:
        Absolute timepoint indices to reconstruct.
    output_dir:
        Root output directory; volumes saved to {output_dir}/estimated_volumes/.
    ref_affine:
        4×4 NIfTI affine for output files.
    batch_size:
        Number of timepoints per forward pass.
    """
    output_dir = Path(output_dir)
    vol_dir = output_dir / "estimated_volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    if ref_affine is None:
        ref_affine = np.eye(4)

    model.eval()

    print(f"Predicting {len(test_indices)} test timepoints → {vol_dir}")
    for i in tqdm(range(0, len(test_indices), batch_size), desc="DeepONet predict"):
        batch_t = test_indices[i : i + batch_size]
        x = torch.tensor(
            branch_inputs[batch_t], dtype=torch.float32
        )  # (batch, n_delay, H, W)
        with torch.no_grad():
            coeffs = model.forward_coefficients(x).numpy()  # (batch, n_pca)

        for j, t in enumerate(batch_t):
            volume = pca.decode(coeffs[j])
            nii = nib.Nifti1Image(volume, ref_affine)
            nib.nifti1.save(nii, str(vol_dir / f"volume_{t}.nii.gz"))

    print("DeepONet prediction complete.")
