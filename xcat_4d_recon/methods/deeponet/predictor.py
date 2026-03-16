"""
Inference for POD-DeepONet: branch input → PCA coefficients → reconstructed volume.
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
    ref_nifti_path: Optional[str] = None,
    batch_size: int = 8,
) -> None:
    """Run DeepONet inference and save reconstructed NIfTI volumes.

    Parameters
    ----------
    model:
        Trained PODDeepONet (on CPU).
    pca:
        Fitted PCAReduction instance.
    branch_inputs:
        Pre-computed branch input array, shape (n_total_aligned, branch_dim).
        Row i corresponds to aligned timepoint index i.
    test_indices:
        Absolute timepoint indices to reconstruct.
    output_dir:
        Root output directory; volumes saved to {output_dir}/estimated_volumes/.
    ref_affine:
        4×4 affine for output NIfTI.
    ref_nifti_path:
        Alternative to ref_affine.
    batch_size:
        Number of timepoints per forward pass.
    """
    output_dir = Path(output_dir)
    vol_dir = output_dir / "estimated_volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    if ref_affine is None:
        if ref_nifti_path:
            ref_affine = nib.load(ref_nifti_path).affine
        else:
            ref_affine = np.eye(4)

    model.eval()
    volume_shape = pca._volume_shape

    print(f"Predicting {len(test_indices)} test timepoints → {vol_dir}")
    for i in tqdm(range(0, len(test_indices), batch_size), desc="DeepONet predict"):
        batch_t = test_indices[i : i + batch_size]
        x = torch.tensor(
            branch_inputs[batch_t], dtype=torch.float32
        )  # (batch, branch_dim)
        with torch.no_grad():
            coeffs = model.forward_coefficients(x).numpy()  # (batch, n_pca)

        for j, t in enumerate(batch_t):
            volume = pca.decode(coeffs[j])
            nii = nib.Nifti1Image(volume, ref_affine)
            nib.nifti1.save(nii, str(vol_dir / f"volume_{t}.nii.gz"))

    print("DeepONet prediction complete.")
