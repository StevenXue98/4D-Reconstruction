"""
Step 4: Benchmark all reconstruction methods.

Loads estimated volumes from outputs/{method}/estimated_volumes/ for each method,
computes RMSE, SSIM, Dice, and centroid displacement for every test timepoint,
writes metrics to outputs/benchmark/metrics.csv and summary.csv, and generates
comparison figures.

Usage:
    python experiments/run_benchmark.py
    python experiments/run_benchmark.py --methods mttde deeponet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.benchmark import run_benchmark
from evaluation.visualize import plot_metric_comparison, plot_loss_curves


def main():
    parser = argparse.ArgumentParser(description="Benchmark reconstruction methods.")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--outputs_dir", default="./outputs")
    parser.add_argument("--benchmark_output_dir", default="./outputs/benchmark")
    parser.add_argument("--n_timepoints", type=int, default=182)
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--voxel_spacing_mm", type=float, nargs=3, default=[1.0, 1.0, 3.0],
                        metavar=("DX", "DY", "DZ"))
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Methods to evaluate (default: all available)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    n_train = int(args.n_timepoints * args.train_frac)
    test_indices = list(range(n_train, args.n_timepoints))
    print(f"Test indices: {test_indices[0]}..{test_indices[-1]} ({len(test_indices)} timepoints)")

    # ── Run benchmark ─────────────────────────────────────────────────────────
    summary = run_benchmark(
        gt_volumes_dir=str(data_dir / "ground_truth" / "volumes"),
        gt_masks_dir=str(data_dir / "ground_truth" / "tumor_masks"),
        outputs_dir=args.outputs_dir,
        test_indices=test_indices,
        voxel_spacing_mm=tuple(args.voxel_spacing_mm),
        methods=args.methods,
        benchmark_output_dir=args.benchmark_output_dir,
    )

    if not summary:
        print("No methods evaluated (outputs not found). Run the reconstruction scripts first.")
        return

    # ── Figures ───────────────────────────────────────────────────────────────
    fig_dir = Path(args.benchmark_output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison(
        summary=summary,
        output_path=str(fig_dir / "metric_comparison.pdf"),
    )

    # Loss curves (if available)
    loss_histories = {}
    mttde_loss_path = Path(args.outputs_dir) / "mttde" / "loss_history.npy"
    deeponet_loss_path = Path(args.outputs_dir) / "deeponet" / "epoch_losses.npy"
    if mttde_loss_path.exists():
        loss_histories["MTTDE"] = np.load(str(mttde_loss_path)).tolist()
    if deeponet_loss_path.exists():
        loss_histories["DeepONet"] = np.load(str(deeponet_loss_path)).tolist()
    if loss_histories:
        plot_loss_curves(
            loss_histories=loss_histories,
            output_path=str(fig_dir / "loss_curves.pdf"),
        )

    print(f"\nAll benchmark outputs in: {args.benchmark_output_dir}")


if __name__ == "__main__":
    main()
