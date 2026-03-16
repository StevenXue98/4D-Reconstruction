"""
Prepare 4D XCAT phantom data for downstream methods.

Adapted from: 4DCT-irregular-motion-main/generateData.py (Huang et al., MICCAI 2024)

Steps:
  1. Extract CT slabs acquired at each timepoint from ground-truth volumes.
  2. Write a text file listing slab paths for SuPReMo.
  3. Sort slabs into 10 respiratory phases.
"""

from __future__ import annotations

import os
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def _sorted_nifti_names(directory: str) -> list[str]:
    names = glob(os.path.join(directory, "*.nii.gz"))
    names.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return names


def prepare_data(
    data_dir: str,
    gt_volumes_dir: str | None = None,
    train_frac: float = 0.80,
) -> None:
    """Extract unsorted CT slabs and sorted 4DCT phases from ground-truth volumes.

    Parameters
    ----------
    data_dir:
        Root data directory (contains rpm_signal.txt, timeIndicesPerSliceAndPhase.txt).
    gt_volumes_dir:
        Directory with per-timepoint NIfTI ground-truth volumes.  Defaults to
        ``{data_dir}/ground_truth/volumes``.
    train_frac:
        Fraction of timepoints used for training (written to dynamic_image_files.txt).
    """
    data_dir = Path(data_dir)
    gt_volumes_dir = Path(gt_volumes_dir) if gt_volumes_dir else data_dir / "ground_truth" / "volumes"

    # ── Load inputs ──────────────────────────────────────────────────────────
    gt_volume_names = _sorted_nifti_names(str(gt_volumes_dir))
    if not gt_volume_names:
        raise FileNotFoundError(f"No NIfTI volumes found in {gt_volumes_dir}")

    time_indices = np.loadtxt(
        data_dir / "timeIndicesPerSliceAndPhase.txt", dtype=int
    )  # shape: (n_slices, n_phases)

    n_timepoints = len(gt_volume_names)
    n_train = int(n_timepoints * train_frac)
    print(f"Found {n_timepoints} volumes. Train split: {n_train}, Test split: {n_timepoints - n_train}")

    # ── Extract unsorted CT slabs ─────────────────────────────────────────────
    slabs_dir = data_dir / "unsort_ct_slabs"
    slabs_dir.mkdir(parents=True, exist_ok=True)
    print("Saving unsorted CT slabs...")
    for i in tqdm(range(n_timepoints)):
        active_slices = np.nonzero(time_indices == i)[0]
        volume = nib.load(gt_volume_names[i])
        affine = volume.affine.copy()
        affine[:, -1] = np.dot(affine, np.array([0, 0, np.min(active_slices), 1]))
        slab = volume.get_fdata()[:, :, active_slices].astype(np.float32)
        nib.nifti1.save(
            nib.Nifti1Image(slab, affine),
            str(slabs_dir / f"slab_{i}.nii.gz"),
        )

    # ── Write dynamic image file list ─────────────────────────────────────────
    dynamic_files_path = data_dir / "dynamic_image_files.txt"
    with open(dynamic_files_path, "w") as f:
        for i in range(n_timepoints):
            f.write(f"{slabs_dir / f'slab_{i}.nii.gz'}\n")
    print(f"Written: {dynamic_files_path}")

    # Also write a train-only file list (for a fair train-only motion model fit)
    train_files_path = data_dir / "dynamic_image_files_train.txt"
    with open(train_files_path, "w") as f:
        for i in range(n_train):
            f.write(f"{slabs_dir / f'slab_{i}.nii.gz'}\n")
    print(f"Written: {train_files_path}")

    # ── Sort slabs into 10 respiratory phases ─────────────────────────────────
    sorted_dir = data_dir / "sorted_4dct"
    sorted_dir.mkdir(parents=True, exist_ok=True)
    affine = nib.load(gt_volume_names[0]).affine
    n_phases = time_indices.shape[1]
    print(f"Sorting slabs into {n_phases} phases...")
    for phase_idx in tqdm(range(n_phases)):
        time_points = np.unique(time_indices[:, phase_idx])
        sorted_vols = [
            nib.load(str(slabs_dir / f"slab_{t}.nii.gz")).get_fdata().astype(np.float32)
            for t in time_points[::-1]
        ]
        nib.nifti1.save(
            nib.Nifti1Image(np.concatenate(sorted_vols, axis=-1), affine),
            str(sorted_dir / f"phase_{phase_idx}.nii.gz"),
        )
    print("Data preparation complete.")
