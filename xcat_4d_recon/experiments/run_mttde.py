"""
Step 3a: Train and evaluate the Measure-Theoretic Time-Delay Embedding method.

Full pipeline:
  1. Load PCA basis.
  2. Extract 1D surrogate signal from 2D projection images (one camera angle).
  3. Estimate time-delay embedding parameters (tau, n) or load from cache.
  4. Build delay-coordinate matrix and encode training volumes.
  5. Build constrained k-means patches.
  6. Train reconstruction network with Wasserstein (energy) loss.
  7. Predict on test timepoints and save NIfTI volumes.

Usage:
    python experiments/run_mttde.py
    python experiments/run_mttde.py --n_iterations 50000 --angle_idx 0
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.pca_reduction import PCAReduction
from methods.mttde.surrogate_extraction import extract_surrogate
from methods.mttde.delay_embedding import compute_embedding_params, build_delay_matrix
from methods.mttde.patching import build_patches
from methods.mttde.trainer import train_mttde
from methods.mttde.predictor import predict_mttde


def _sorted_paths(directory: str, prefix: str, ext: str = ".nii.gz") -> list[str]:
    paths = glob(str(Path(directory) / f"{prefix}*.{ext.lstrip('.')}"))
    paths.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return paths


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MTTDE.")
    # Paths
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--artifacts_dir", default="./artifacts")
    parser.add_argument("--output_dir", default="./outputs/mttde")
    # Dataset
    parser.add_argument("--n_timepoints", type=int, default=182)
    parser.add_argument("--train_frac", type=float, default=0.80)
    # PCA
    parser.add_argument("--n_components", type=int, default=64)
    # Surrogate — projection-based
    parser.add_argument("--projections_dir", default=None,
                        help="Directory containing projection .npy files "
                             "(default: <data_dir>/projections)")
    parser.add_argument("--angle_idx", type=int, default=0,
                        help="Which camera angle to use for surrogate extraction (default: 0)")
    # Embedding
    parser.add_argument("--tau", type=int, default=None, help="Override delay tau")
    parser.add_argument("--n_embed", type=int, default=None, help="Override embedding dim n")
    parser.add_argument("--tau_max", type=int, default=20)
    parser.add_argument("--n_max", type=int, default=10)
    # Patches
    parser.add_argument("--n_patches", type=int, default=20)
    # Network
    parser.add_argument("--hidden_dim", type=int, default=100)
    parser.add_argument("--n_hidden_layers", type=int, default=4)
    # Training
    parser.add_argument("--n_iterations", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--loss_type", default="energy", choices=["energy", "sinkhorn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    projections_dir = Path(args.projections_dir) if args.projections_dir else data_dir / "projections"
    n_train = int(args.n_timepoints * args.train_frac)
    test_indices = list(range(n_train, args.n_timepoints))
    print(f"Train: 0..{n_train-1}  Test: {n_train}..{args.n_timepoints-1}")

    # ── 1. Load PCA ───────────────────────────────────────────────────────────
    pca = PCAReduction(n_components=args.n_components)
    pca.load(str(artifacts_dir / "pca_basis.npz"))

    # ── 2. Extract surrogate signal from projection images ────────────────────
    proj_paths = sorted(
        projections_dir.glob(f"proj_*_angle_{args.angle_idx:02d}.npy"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not proj_paths:
        raise FileNotFoundError(
            f"No projection files found in {projections_dir} for angle_idx={args.angle_idx}.\n"
            "Run: python experiments/prepare_data.py"
        )
    proj_paths = [str(p) for p in proj_paths]
    print(f"\nExtracting 1D surrogate from {len(proj_paths)} projections "
          f"(camera angle index {args.angle_idx}) ...")
    surrogate = extract_surrogate(proj_paths)
    print(f"Surrogate signal shape: {surrogate.shape}")

    scale = float(np.max(np.abs(surrogate)))
    surrogate_norm = surrogate / scale if scale > 0 else surrogate.copy()

    # ── 3. Compute embedding parameters ──────────────────────────────────────
    delay_params_file = str(artifacts_dir / "delay_params.json")
    tau, n_embed = compute_embedding_params(
        signal=surrogate_norm[:n_train],
        tau_override=args.tau,
        n_override=args.n_embed,
        tau_max=args.tau_max,
        n_max=args.n_max,
        delay_params_file=delay_params_file,
    )
    print(f"Using tau={tau}, n={n_embed}")

    # ── 4. Build delay matrix and encode training volumes ─────────────────────
    print("\nBuilding delay matrix ...")
    delay_matrix_full = build_delay_matrix(surrogate_norm, tau, n_embed)
    trim = (n_embed - 1) * tau
    train_delay = delay_matrix_full[: n_train - trim]

    print("Encoding training volumes with PCA ...")
    gt_volumes_dir = data_dir / "ground_truth" / "volumes"
    gt_paths = _sorted_paths(str(gt_volumes_dir), "volume_")
    if not gt_paths:
        raise FileNotFoundError(f"No GT volumes in {gt_volumes_dir}")

    train_gt_paths = gt_paths[trim:n_train]
    pca_coefficients = pca.encode_many(train_gt_paths)
    print(f"PCA coefficient matrix: {pca_coefficients.shape}")

    # ── 5. Build patches ──────────────────────────────────────────────────────
    patch_inputs, patch_outputs = build_patches(
        delay_coords=train_delay,
        pca_coefficients=pca_coefficients,
        n_patches=args.n_patches,
        patches_file=str(artifacts_dir / "mttde_patches.pkl"),
    )

    # ── 6. Train ──────────────────────────────────────────────────────────────
    net, loss_history = train_mttde(
        patch_inputs=patch_inputs,
        patch_outputs=patch_outputs,
        input_dim=n_embed,
        output_dim=pca.n_components,
        hidden_dim=args.hidden_dim,
        n_hidden_layers=args.n_hidden_layers,
        loss_type=args.loss_type,
        n_iterations=args.n_iterations,
        learning_rate=args.learning_rate,
        checkpoint_dir=str(output_dir / "checkpoints"),
        seed=args.seed,
        device=args.device,
    )
    np.save(str(output_dir / "loss_history.npy"), np.array(loss_history))

    # ── 7. Predict on test split ──────────────────────────────────────────────
    predict_mttde(
        net=net,
        pca=pca,
        surrogate_signal=surrogate_norm,
        tau=tau,
        n=n_embed,
        test_indices=test_indices,
        output_dir=str(output_dir),
        ref_affine=nib.load(gt_paths[0]).affine,
    )

    print("\nMTTDE pipeline complete.")


if __name__ == "__main__":
    main()
