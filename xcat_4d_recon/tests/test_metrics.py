"""Tests for evaluation metrics."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import (
    compute_mse,
    compute_ssim,
    compute_dice,
    compute_centroid_displacement,
    evaluate_timepoint,
)


def test_mse_identical():
    vol = np.random.rand(10, 10, 10).astype(np.float32)
    assert compute_mse(vol, vol) == pytest.approx(0.0, abs=1e-7)


def test_mse_known():
    gt = np.zeros((4, 4, 4), dtype=np.float32)
    pred = np.ones((4, 4, 4), dtype=np.float32)
    assert compute_mse(gt, pred) == pytest.approx(1.0, rel=1e-5)


def test_ssim_identical():
    vol = np.random.rand(20, 20, 20).astype(np.float32) * 1000
    ssim_val = compute_ssim(vol, vol)
    assert ssim_val == pytest.approx(1.0, abs=1e-5)


def test_ssim_range():
    a = np.random.rand(20, 20, 20).astype(np.float32)
    b = np.random.rand(20, 20, 20).astype(np.float32)
    ssim_val = compute_ssim(a, b)
    assert -1.0 <= ssim_val <= 1.0


def test_dice_perfect():
    mask = np.zeros((10, 10, 10), dtype=np.float32)
    mask[3:7, 3:7, 3:7] = 1.0
    assert compute_dice(mask, mask) == pytest.approx(1.0)


def test_dice_no_overlap():
    a = np.zeros((10, 10, 10), dtype=np.float32)
    b = np.zeros((10, 10, 10), dtype=np.float32)
    a[0:3, 0:3, 0:3] = 1.0
    b[7:10, 7:10, 7:10] = 1.0
    assert compute_dice(a, b) == pytest.approx(0.0)


def test_dice_empty():
    empty = np.zeros((10, 10, 10), dtype=np.float32)
    assert compute_dice(empty, empty) == pytest.approx(0.0)


def test_centroid_displacement_zero():
    mask = np.zeros((20, 20, 20), dtype=np.float32)
    mask[8:12, 8:12, 8:12] = 1.0
    disp = compute_centroid_displacement(mask, mask, voxel_spacing_mm=(1.0, 1.0, 1.0))
    assert disp == pytest.approx(0.0, abs=1e-5)


def test_centroid_displacement_known():
    gt = np.zeros((20, 20, 20), dtype=np.float32)
    pred = np.zeros((20, 20, 20), dtype=np.float32)
    gt[9:11, 9:11, 9:11] = 1.0
    pred[9:11, 9:11, 12:14] = 1.0  # shifted by 3 in z
    disp = compute_centroid_displacement(gt, pred, voxel_spacing_mm=(1.0, 1.0, 1.0))
    assert disp == pytest.approx(3.0, rel=0.1)


def test_centroid_displacement_nan_empty():
    empty = np.zeros((10, 10, 10), dtype=np.float32)
    mask = np.zeros((10, 10, 10), dtype=np.float32)
    mask[4:6, 4:6, 4:6] = 1.0
    assert np.isnan(compute_centroid_displacement(empty, mask))


def test_evaluate_timepoint_keys():
    vol = np.random.rand(10, 10, 10).astype(np.float32)
    mask = (vol > 0.5).astype(np.float32)
    result = evaluate_timepoint(vol, vol, mask, mask)
    assert "rmse" in result
    assert "ssim" in result
    assert "dice" in result
    assert "centroid_mm" in result
