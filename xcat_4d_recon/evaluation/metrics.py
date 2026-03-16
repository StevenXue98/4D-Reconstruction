"""
Quantitative evaluation metrics for 4D CT reconstruction quality.

Adapted from: 4DCT-irregular-motion-main/evaluation.py (Huang et al., MICCAI 2024)

Metrics:
  - RMSE : root mean squared error between estimated and ground-truth HU values.
  - SSIM : structural similarity index (perceptual quality proxy).
  - Dice : volumetric overlap of tumour masks.
  - Centroid displacement : Euclidean distance between tumour centroids (mm).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.measure import label, regionprops


def compute_mse(gt: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error between ground-truth and predicted volumes.

    Matches the metric in the 4DCT reference evaluation.py:
      ``np.mean(np.sqrt(mse))``  where mse = np.nanmean(np.square(gt - pred)).

    Parameters
    ----------
    gt, pred : shape (x, y, z), float32

    Returns
    -------
    rmse : float
    """
    return float(np.sqrt(np.nanmean(np.square(gt.astype(np.float64) - pred.astype(np.float64)))))


def compute_ssim(gt: np.ndarray, pred: np.ndarray) -> float:
    """Structural Similarity Index (SSIM).

    Parameters
    ----------
    gt, pred : shape (x, y, z)

    Returns
    -------
    ssim_val : float in [-1, 1]
    """
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise ImportError("scikit-image is required for SSIM.  pip install scikit-image") from exc

    data_range = float(gt.max() - gt.min())
    if data_range == 0:
        return 1.0
    ssim_val, _ = structural_similarity(
        gt.astype(np.float64),
        pred.astype(np.float64),
        data_range=data_range,
        full=True,
    )
    return float(ssim_val)


def compute_dice(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Sørensen-Dice coefficient for binary tumour mask overlap.

    Matches the reference:
      ``2 * sum(gt * pred) / (sum(gt) + sum(pred))``

    Parameters
    ----------
    gt_mask   : shape (x, y, z), binary (0/1) or float.
    pred_mask : shape (x, y, z), binary or float.  Thresholded at ``threshold``.
    threshold : binarisation threshold for pred_mask.

    Returns
    -------
    dice : float in [0, 1]  (returns 0.0 if both masks are empty)
    """
    gt = gt_mask.astype(bool)
    pred = (pred_mask > threshold).astype(bool)
    intersection = np.sum(gt & pred)
    denom = np.sum(gt) + np.sum(pred)
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


def compute_centroid_displacement(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 3.0),
    threshold: float = 0.5,
) -> float:
    """Euclidean displacement between tumour centroids in mm.

    Matches the reference:
      centroid from ``regionprops(label(mask))[0].centroid``,
      gap = sqrt(sum((c_pred - c_gt)^2)).

    Parameters
    ----------
    gt_mask, pred_mask : shape (x, y, z)
    voxel_spacing_mm   : (dx, dy, dz) in mm; default XCAT: (1, 1, 3)
    threshold          : binarisation threshold for predicted mask

    Returns
    -------
    displacement_mm : float  (returns NaN if either mask is empty)
    """
    gt_bin = (gt_mask > threshold).astype(np.uint8)
    pred_bin = (pred_mask > threshold).astype(np.uint8)

    gt_props = regionprops(label(gt_bin))
    pred_props = regionprops(label(pred_bin))

    if not gt_props or not pred_props:
        return float("nan")

    gt_centroid = np.array(gt_props[0].centroid)    # voxel coords
    pred_centroid = np.array(pred_props[0].centroid)

    # Convert to mm
    spacing = np.array(voxel_spacing_mm, dtype=np.float64)
    gap_vox = gt_centroid - pred_centroid
    gap_mm = gap_vox * spacing
    return float(np.sqrt(np.sum(gap_mm ** 2)))


def evaluate_timepoint(
    gt_volume: np.ndarray,
    pred_volume: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
    pred_mask: Optional[np.ndarray] = None,
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 3.0),
) -> dict[str, float]:
    """Compute all metrics for a single timepoint.

    Returns
    -------
    dict with keys: rmse, ssim, dice, centroid_mm
    """
    result: dict[str, float] = {
        "rmse": compute_mse(gt_volume, pred_volume),
        "ssim": compute_ssim(gt_volume, pred_volume),
    }
    if gt_mask is not None and pred_mask is not None:
        result["dice"] = compute_dice(gt_mask, pred_mask)
        result["centroid_mm"] = compute_centroid_displacement(
            gt_mask, pred_mask, voxel_spacing_mm
        )
    return result
