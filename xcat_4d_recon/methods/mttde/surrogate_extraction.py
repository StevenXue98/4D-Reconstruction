"""
Extract a 1D surrogate respiratory signal from acquired CT slabs.

The surrogate signal is the observable used to build the time-delay embedding.
It must be extractable from the *acquired* (unsorted) CT slabs at test time so
there is no data leakage from ground-truth volumes.

Two extraction strategies:
  pca_mode  – project each CT slab onto PCA component 0 (already captures dominant
               respiratory motion from the training-fitted PCA basis).
  mean_hu   – mean Hounsfield unit inside a user-specified lung bounding box.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm


def extract_surrogate(
    slab_paths: list[str],
    method: str = "pca_mode",
    pca_mean: Optional[np.ndarray] = None,
    pca_component0: Optional[np.ndarray] = None,
    volume_shape: Optional[tuple[int, ...]] = None,
    lung_bbox: Optional[list[int]] = None,
) -> np.ndarray:
    """Extract a 1-D surrogate signal from CT slabs.

    Parameters
    ----------
    slab_paths:
        Ordered list of CT slab NIfTI paths.
    method:
        ``"pca_mode"`` or ``"mean_hu"``.
    pca_mean:
        Flattened per-voxel mean (required for ``"pca_mode"``).
    pca_component0:
        First PCA component, shape (n_voxels,) (required for ``"pca_mode"``).
    volume_shape:
        Expected full-volume shape (x, y, z); slab values outside bounds are
        zero-padded before projecting.
    lung_bbox:
        [x0, x1, y0, y1, z0, z1] voxel indices (required for ``"mean_hu"``).

    Returns
    -------
    signal : shape (n_timepoints,)
    """
    signal = []
    for slab_path in tqdm(slab_paths, desc="Extracting surrogate", leave=False):
        slab_img = nib.load(slab_path)
        slab_data = slab_img.get_fdata().astype(np.float32)

        if method == "pca_mode":
            if pca_mean is None or pca_component0 is None or volume_shape is None:
                raise ValueError("pca_mode requires pca_mean, pca_component0, and volume_shape.")
            # Zero-pad slab to full volume shape along z-axis
            full = np.zeros(volume_shape, dtype=np.float32)
            # Determine which z-slices this slab covers from affine
            affine = slab_img.affine
            z0_mm = affine[2, 3]
            z_spacing = affine[2, 2] if affine[2, 2] != 0 else 1.0
            # Use origin z in voxel coords based on affine
            origin_vox = int(round(z0_mm / z_spacing))
            z_start = max(0, origin_vox)
            z_end = min(volume_shape[2], z_start + slab_data.shape[2])
            slab_z_end = z_end - z_start
            if slab_z_end > 0:
                full[:slab_data.shape[0], :slab_data.shape[1], z_start:z_end] = slab_data[:, :, :slab_z_end]
            flat = full.ravel() - pca_mean
            projection = float(np.dot(flat, pca_component0))
            signal.append(projection)

        elif method == "mean_hu":
            if lung_bbox is None:
                raise ValueError("mean_hu requires lung_bbox=[x0,x1,y0,y1,z0,z1].")
            x0, x1, y0, y1, z0, z1 = lung_bbox
            # Intersect bbox with slab's actual z-extent
            slab_z = slab_data.shape[2]
            z_end = min(z1, slab_z)
            if z_end > z0 and x1 > x0 and y1 > y0:
                roi = slab_data[x0:x1, y0:y1, z0:z_end]
                signal.append(float(roi.mean()))
            else:
                signal.append(0.0)
        else:
            raise ValueError(f"Unknown surrogate extraction method: {method!r}")

    return np.array(signal, dtype=np.float64)
