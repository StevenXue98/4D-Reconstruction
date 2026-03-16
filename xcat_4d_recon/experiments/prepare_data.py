"""
Step 1: Data preparation.

Extracts unsorted CT slabs from ground-truth XCAT volumes, generates surrogate
respiratory signals, writes file lists for SuPReMo, and (optionally) generates
simulated 2D projection images from the full 3D volumes to mimic the fusion
camera observation model.

Usage:
    python experiments/prepare_data.py
    python experiments/prepare_data.py --data_dir /path/to/data --train_frac 0.8
    python experiments/prepare_data.py --n_angles 8 --angle_range 360
    python experiments/prepare_data.py --skip_projections
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.prepare_data import prepare_data
from preprocessing.generate_surrogates import generate_surrogates
from preprocessing.generate_projections import generate_projections
from preprocessing.prepare_data import _sorted_nifti_names


def main():
    parser = argparse.ArgumentParser(description="Prepare 4D XCAT data for all methods.")
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument("--gt_volumes_dir", default=None, help="Ground-truth volumes dir")
    parser.add_argument("--train_frac", type=float, default=0.80,
                        help="Fraction of timepoints for training split")
    # Projection arguments
    parser.add_argument("--skip_projections", action="store_true",
                        help="Skip 2D projection generation")
    parser.add_argument("--n_angles", type=int, default=6,
                        help="Number of camera angles for projection generation (default: 6)")
    parser.add_argument("--angle_start", type=float, default=0.0,
                        help="Starting angle in degrees (default: 0)")
    parser.add_argument("--angle_range", type=float, default=180.0,
                        help="Total angular span in degrees (default: 180)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    gt_volumes_dir = (
        Path(args.gt_volumes_dir) if args.gt_volumes_dir
        else data_dir / "ground_truth" / "volumes"
    )

    print("=" * 60)
    print("Step 1a: Extracting CT slabs and sorting phases")
    print("=" * 60)
    prepare_data(
        data_dir=str(data_dir),
        gt_volumes_dir=str(gt_volumes_dir),
        train_frac=args.train_frac,
    )

    print("\n" + "=" * 60)
    print("Step 1b: Generating surrogate signals")
    print("=" * 60)
    generate_surrogates(data_dir=str(data_dir))

    if not args.skip_projections:
        print("\n" + "=" * 60)
        print(f"Step 1c: Generating 2D projections ({args.n_angles} angles, "
              f"{args.angle_start}–{args.angle_start + args.angle_range}°)")
        print("=" * 60)
        volume_paths = _sorted_nifti_names(str(gt_volumes_dir))
        generate_projections(
            volume_paths=volume_paths,
            output_dir=str(data_dir / "projections"),
            n_angles=args.n_angles,
            angle_start=args.angle_start,
            angle_range=args.angle_range,
        )
    else:
        print("\nStep 1c: Skipping projection generation (--skip_projections set)")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
