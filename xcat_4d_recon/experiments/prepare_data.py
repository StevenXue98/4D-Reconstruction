"""
Step 1: Data preparation.

Extracts unsorted CT slabs from ground-truth XCAT volumes,
generates surrogate respiratory signals, and writes file lists for SuPReMo.

Usage:
    python experiments/prepare_data.py
    python experiments/prepare_data.py --data_dir /path/to/data --train_frac 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.prepare_data import prepare_data
from preprocessing.generate_surrogates import generate_surrogates


def main():
    parser = argparse.ArgumentParser(description="Prepare 4D XCAT data for all methods.")
    parser.add_argument("--data_dir", default="./data", help="Root data directory")
    parser.add_argument("--gt_volumes_dir", default=None, help="Ground-truth volumes dir")
    parser.add_argument("--train_frac", type=float, default=0.80,
                        help="Fraction of timepoints for training split")
    args = parser.parse_args()

    print("=" * 60)
    print("Step 1a: Extracting CT slabs and sorting phases")
    print("=" * 60)
    prepare_data(
        data_dir=args.data_dir,
        gt_volumes_dir=args.gt_volumes_dir,
        train_frac=args.train_frac,
    )

    print("\n" + "=" * 60)
    print("Step 1b: Generating surrogate signals")
    print("=" * 60)
    generate_surrogates(data_dir=args.data_dir)

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
