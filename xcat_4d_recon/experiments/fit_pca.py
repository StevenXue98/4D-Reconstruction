"""
Step 2: Fit and save PCA basis on the training split.

Must be run before MTTDE and DeepONet training.

Usage:
    python experiments/fit_pca.py
    python experiments/fit_pca.py --n_components 64 --data_dir ./data
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.pca_reduction import PCAReduction


def _sorted_volume_paths(gt_volumes_dir: str) -> list[str]:
    names = glob(str(Path(gt_volumes_dir) / "*.nii.gz"))
    names.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return names


def main():
    parser = argparse.ArgumentParser(description="Fit PCA on training volumes.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--gt_volumes_dir", default=None)
    parser.add_argument("--n_components", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--artifacts_dir", default="./artifacts")
    args = parser.parse_args()

    gt_volumes_dir = args.gt_volumes_dir or str(Path(args.data_dir) / "ground_truth" / "volumes")
    all_paths = _sorted_volume_paths(gt_volumes_dir)
    if not all_paths:
        raise FileNotFoundError(f"No NIfTI volumes found in {gt_volumes_dir}")

    n_train = int(len(all_paths) * args.train_frac)
    train_paths = all_paths[:n_train]
    print(f"Fitting PCA on {n_train}/{len(all_paths)} training volumes ...")

    pca = PCAReduction(n_components=args.n_components, batch_size=args.batch_size)
    normalisation_file = str(Path(args.artifacts_dir) / "normalisation.json")
    pca.fit(train_paths, normalisation_file=normalisation_file)

    basis_file = str(Path(args.artifacts_dir) / "pca_basis.npz")
    pca.save(basis_file)
    print(f"\nPCA basis saved → {basis_file}")

    # Also fit PCA for tumour masks if available
    gt_masks_dir = str(Path(args.data_dir) / "ground_truth" / "tumor_masks")
    mask_paths = sorted(
        glob(str(Path(gt_masks_dir) / "*.nii.gz")),
        key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]),
    )
    if mask_paths:
        train_mask_paths = mask_paths[:n_train]
        print(f"\nFitting PCA on {n_train} tumour masks ...")
        pca_mask = PCAReduction(n_components=args.n_components, batch_size=args.batch_size)
        pca_mask.fit(train_mask_paths)
        mask_basis_file = str(Path(args.artifacts_dir) / "pca_mask_basis.npz")
        pca_mask.save(mask_basis_file)
        print(f"Mask PCA basis saved → {mask_basis_file}")
    else:
        print("No tumour masks found; skipping mask PCA.")


if __name__ == "__main__":
    main()
