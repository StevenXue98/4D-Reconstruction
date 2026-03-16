"""
Inference for MTTDE: map delay-coordinate vectors → reconstructed volumes.

Pipeline:
  1. Build delay-coordinate vector for test timepoint t using the 1D surrogate.
  2. Forward pass through trained ReconstructionNet → PCA coefficients.
  3. PCA decode → 3D volume array.
  4. Save as NIfTI (reusing affine from a reference volume).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .delay_embedding import build_delay_matrix
from .network import ReconstructionNet
from preprocessing.pca_reduction import PCAReduction


def predict_mttde(
    net: ReconstructionNet,
    pca: PCAReduction,
    surrogate_signal: np.ndarray,
    tau: int,
    n: int,
    test_indices: list[int],
    output_dir: str,
    ref_affine: Optional[np.ndarray] = None,
    ref_nifti_path: Optional[str] = None,
    scale: Optional[float] = None,
) -> None:
    """Run MTTDE inference on test timepoints and save reconstructed NIfTI volumes.

    Parameters
    ----------
    net:
        Trained ReconstructionNet (on CPU).
    pca:
        Fitted PCAReduction instance.
    surrogate_signal:
        1-D surrogate signal for *all* timepoints (train + test), shape (T,).
    tau:
        Time delay used during training.
    n:
        Embedding dimension used during training.
    test_indices:
        List of absolute timepoint indices to reconstruct (e.g., [146..181]).
    output_dir:
        Directory to write reconstructed NIfTI volumes.
    ref_affine:
        4×4 affine matrix for the output NIfTI.
    ref_nifti_path:
        Alternative to ref_affine: load affine from this NIfTI file.
    scale:
        If the surrogate signal was normalised during training (divide by max),
        pass the same scale factor here.
    """
    output_dir = Path(output_dir)
    (output_dir / "estimated_volumes").mkdir(parents=True, exist_ok=True)

    # Affine for output NIfTI
    if ref_affine is None:
        if ref_nifti_path:
            ref_affine = nib.load(ref_nifti_path).affine
        else:
            ref_affine = np.eye(4)

    # Normalise surrogate the same way as during training
    if scale is not None:
        signal = surrogate_signal / scale
    else:
        signal = surrogate_signal.copy()

    net.eval()
    trim = (n - 1) * tau

    print(f"Predicting {len(test_indices)} test timepoints → {output_dir / 'estimated_volumes'}")
    for t in tqdm(test_indices, desc="MTTDE predict"):
        if t < trim:
            print(f"  [warn] timepoint {t} < trim={trim}; skipping.")
            continue
        # Build delay vector for this timepoint
        delay_vec = np.array(
            [signal[t - j * tau] for j in range(n)], dtype=np.float32
        )  # shape (n,)
        x = torch.tensor(delay_vec, dtype=torch.float32).unsqueeze(0)  # (1, n)
        with torch.no_grad():
            coeff_pred = net(x).squeeze(0).numpy()  # (n_pca,)

        volume = pca.decode(coeff_pred)  # (x, y, z)
        nii = nib.Nifti1Image(volume, ref_affine)
        out_path = output_dir / "estimated_volumes" / f"volume_{t}.nii.gz"
        nib.nifti1.save(nii, str(out_path))

    print("MTTDE prediction complete.")
