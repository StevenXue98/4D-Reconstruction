"""
Benchmark all reconstruction methods against ground-truth XCAT volumes.

Loads estimated volumes from outputs/{method}/estimated_volumes/ and masks from
outputs/{method}/estimated_tumormasks/, computes metrics for every test timepoint,
and writes a consolidated CSV plus summary statistics.
"""

from __future__ import annotations

import csv
import sys
from glob import glob
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from tqdm import tqdm

# Allow running as a script directly (python evaluation/benchmark.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.metrics import evaluate_timepoint


# Registry of method name → relative output subdirectory
METHODS = {
    "surrogate_driven": "surr_driven",
    "surrogate_free": "surr_free",
    "surrogate_optimized": "surr_optimized",
    "mttde": "mttde",
    "deeponet": "deeponet",
}


def _load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata().astype(np.float32)


def _sorted_nifti_names(directory: str) -> list[str]:
    names = glob(str(Path(directory) / "*.nii.gz"))
    names.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return names


def run_benchmark(
    gt_volumes_dir: str,
    gt_masks_dir: str,
    outputs_dir: str,
    test_indices: list[int],
    voxel_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 3.0),
    methods: Optional[list[str]] = None,
    benchmark_output_dir: str = "./outputs/benchmark",
) -> dict[str, dict[str, float]]:
    """Evaluate all methods and save results.

    Parameters
    ----------
    gt_volumes_dir:
        Path to ground-truth volume NIfTIs (volume_{t}.nii.gz).
    gt_masks_dir:
        Path to ground-truth mask NIfTIs (mask_{t}.nii.gz or volume_{t}.nii.gz).
    outputs_dir:
        Root outputs directory containing per-method subdirectories.
    test_indices:
        List of timepoint indices in the test split.
    voxel_spacing_mm:
        Voxel spacing in mm (x, y, z) for centroid displacement.
    methods:
        Subset of method names to evaluate.  None = all registered methods.
    benchmark_output_dir:
        Directory for CSV and figure outputs.

    Returns
    -------
    summary : {method: {metric: mean_value}}
    """
    benchmark_output_dir = Path(benchmark_output_dir)
    benchmark_output_dir.mkdir(parents=True, exist_ok=True)

    methods_to_run = methods or list(METHODS.keys())

    # ── CSV header ────────────────────────────────────────────────────────────
    csv_path = benchmark_output_dir / "metrics.csv"
    fieldnames = ["method", "timepoint", "rmse", "ssim", "dice", "centroid_mm"]
    rows: list[dict] = []

    for method_name in methods_to_run:
        subdir = METHODS.get(method_name)
        if subdir is None:
            print(f"[benchmark] Unknown method {method_name!r}; skipping.")
            continue

        vol_dir = Path(outputs_dir) / subdir / "estimated_volumes"
        mask_dir = Path(outputs_dir) / subdir / "estimated_tumormasks"
        has_masks = mask_dir.exists() and len(list(mask_dir.glob("*.nii.gz"))) > 0

        if not vol_dir.exists():
            print(f"[benchmark] Outputs not found for {method_name}: {vol_dir}; skipping.")
            continue

        print(f"\nEvaluating {method_name} ...")
        method_rows: list[dict] = []

        for t in tqdm(test_indices, desc=method_name):
            gt_vol_path = Path(gt_volumes_dir) / f"volume_{t}.nii.gz"
            pred_vol_path = vol_dir / f"volume_{t}.nii.gz"
            if not gt_vol_path.exists() or not pred_vol_path.exists():
                continue

            gt_vol = _load_nifti(str(gt_vol_path))
            pred_vol = _load_nifti(str(pred_vol_path))

            gt_mask, pred_mask = None, None
            if has_masks:
                gt_mask_path = Path(gt_masks_dir) / f"mask_{t}.nii.gz"
                pred_mask_path = mask_dir / f"mask_{t}.nii.gz"
                if gt_mask_path.exists() and pred_mask_path.exists():
                    gt_mask = _load_nifti(str(gt_mask_path))
                    pred_mask = _load_nifti(str(pred_mask_path))

            m = evaluate_timepoint(gt_vol, pred_vol, gt_mask, pred_mask, voxel_spacing_mm)
            row = {"method": method_name, "timepoint": t, **m}
            # Fill missing keys
            for k in ["dice", "centroid_mm"]:
                if k not in row:
                    row[k] = float("nan")
            method_rows.append(row)

        rows.extend(method_rows)

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nMetrics CSV → {csv_path}")

    # ── Summary statistics ────────────────────────────────────────────────────
    summary: dict[str, dict[str, float]] = {}
    current_method_rows: dict[str, list[dict]] = {}
    for row in rows:
        current_method_rows.setdefault(row["method"], []).append(row)

    print("\n" + "=" * 70)
    print(f"{'Method':<22} {'RMSE':>10} {'SSIM':>10} {'Dice':>10} {'Centroid (mm)':>14}")
    print("-" * 70)
    for method_name, mrows in current_method_rows.items():
        means: dict[str, float] = {}
        for metric in ["rmse", "ssim", "dice", "centroid_mm"]:
            vals = [r[metric] for r in mrows if not np.isnan(r.get(metric, float("nan")))]
            means[metric] = float(np.mean(vals)) if vals else float("nan")
        summary[method_name] = means
        print(
            f"{method_name:<22} "
            f"{means['rmse']:>10.4f} "
            f"{means['ssim']:>10.4f} "
            f"{means['dice']:>10.4f} "
            f"{means['centroid_mm']:>14.4f}"
        )
    print("=" * 70)

    # Save summary
    summary_path = benchmark_output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "rmse", "ssim", "dice", "centroid_mm"])
        writer.writeheader()
        for method_name, means in summary.items():
            writer.writerow({"method": method_name, **means})
    print(f"Summary CSV → {summary_path}")

    return summary
