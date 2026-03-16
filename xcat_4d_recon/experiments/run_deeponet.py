"""
Step 3b: Train and evaluate the POD-DeepONet reconstruction method.

Full pipeline:
  1. Downsample 2D projection images to (proj_n × proj_n) via Gaussian-smoothed
     resize.  proj_n controls the tradeoff between training speed, denoising,
     and spatial detail retained in the branch input.
  2. Extract 1D surrogate from projections to determine Takens τ and n.
  3. Build branch inputs: for each timepoint t, stack n downsampled projection
     images at delays [t, t-τ, ..., t-(n-1)τ] → (n, proj_n, proj_n) tensor.
  4. Encode training volumes with PCA (POD trunk basis).
  5. Train CNN branch: (n, proj_n, proj_n) → n_pca coefficients, MSE loss.
  6. Predict on test timepoints (optionally at full resolution) → NIfTI volumes.

Resolution invariance
  The CNN branch uses AdaptiveAvgPool2d and therefore accepts any (H, W) input.
  To run inference at full resolution, set --inference_full_res.

Usage:
    python experiments/run_deeponet.py
    python experiments/run_deeponet.py --proj_n 20 --n_epochs 500
    python experiments/run_deeponet.py --proj_n 15 --inference_full_res
"""

from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.pca_reduction import PCAReduction
from preprocessing.generate_projections import downsample_projections
from methods.mttde.surrogate_extraction import extract_surrogate
from methods.mttde.delay_embedding import compute_embedding_params
from methods.deeponet.dataset import ProjectionDataset
from methods.deeponet.pod_deeponet import PODDeepONet
from methods.deeponet.trainer import train_deeponet
from methods.deeponet.predictor import predict_deeponet


def _sorted_paths(directory: str, prefix: str) -> list[str]:
    paths = glob(str(Path(directory) / f"{prefix}*.nii.gz"))
    paths.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return paths


def _load_proj_stack(
    proj_dir: Path,
    t: int,
    tau: int,
    n_delay: int,
    angle_idx: int,
    prefix: str = "proj_small_",
) -> np.ndarray:
    """Load n_delay projection images for timepoint t → (n_delay, H, W)."""
    stack = []
    for j in range(n_delay):
        vol_idx = t - j * tau
        path = proj_dir / f"{prefix}{vol_idx:03d}_angle_{angle_idx:02d}.npy"
        stack.append(np.load(str(path)).astype(np.float32))
    return np.stack(stack, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate POD-DeepONet.")
    # Paths
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--artifacts_dir", default="./artifacts")
    parser.add_argument("--output_dir", default="./outputs/deeponet")
    # Dataset
    parser.add_argument("--n_timepoints", type=int, default=182)
    parser.add_argument("--train_frac", type=float, default=0.80)
    # PCA
    parser.add_argument("--n_components", type=int, default=64)
    # Projection / observation
    parser.add_argument("--projections_dir", default=None,
                        help="Directory of full-res projection .npy files "
                             "(default: <data_dir>/projections)")
    parser.add_argument("--angle_idx", type=int, default=0,
                        help="Camera angle index to use (default: 0)")
    # Downsampling resolution — key tradeoff knob
    parser.add_argument("--proj_n", type=int, default=15,
                        help="Projection is downsampled to (proj_n × proj_n) before "
                             "feeding to the branch CNN.  Larger = more spatial detail "
                             "but slower training.  (default: 15)")
    # Inference resolution
    parser.add_argument("--inference_full_res", action="store_true",
                        help="At inference, pass full-resolution projections to the CNN "
                             "instead of the downsampled ones.  The CNN's adaptive "
                             "pooling handles any input size.")
    # Delay embedding (cached from MTTDE run if available)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--n_embed", type=int, default=None)
    # CNN architecture
    parser.add_argument("--base_channels", type=int, default=32,
                        help="Feature maps in first conv layer (default: 32)")
    parser.add_argument("--pool_size", type=int, default=4,
                        help="AdaptiveAvgPool2d output size (default: 4)")
    # Training
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", default="cosine",
                        choices=["cosine", "step", "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    projections_dir = Path(args.projections_dir) if args.projections_dir else data_dir / "projections"
    proj_small_dir = projections_dir / f"small_{args.proj_n}x{args.proj_n}"

    n_train = int(args.n_timepoints * args.train_frac)
    test_indices = list(range(n_train, args.n_timepoints))
    print(f"Train: 0..{n_train-1}  Test: {n_train}..{args.n_timepoints-1}")

    # ── 1. Downsample projections ─────────────────────────────────────────────
    if not proj_small_dir.exists() or not any(proj_small_dir.glob("proj_small_*.npy")):
        print(f"\nDownsampling projections to {args.proj_n}×{args.proj_n} ...")
        downsample_projections(
            input_dir=str(projections_dir),
            output_dir=str(proj_small_dir),
            target_h=args.proj_n,
            target_w=args.proj_n,
        )
    else:
        print(f"Downsampled projections ({args.proj_n}×{args.proj_n}) already exist, skipping.")

    # ── 2. Estimate Takens τ and n ────────────────────────────────────────────
    proj_paths = sorted(
        projections_dir.glob(f"proj_*_angle_{args.angle_idx:02d}.npy"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not proj_paths:
        raise FileNotFoundError(
            f"No projection files found in {projections_dir} for angle_idx={args.angle_idx}.\n"
            "Run: python experiments/prepare_data.py"
        )
    print(f"\nExtracting 1D surrogate from {len(proj_paths)} full-res projections ...")
    surrogate = extract_surrogate([str(p) for p in proj_paths])
    scale = float(np.max(np.abs(surrogate)))
    surrogate_norm = surrogate / scale if scale > 0 else surrogate.copy()

    delay_params_file = str(artifacts_dir / "delay_params.json")
    tau, n_embed = compute_embedding_params(
        signal=surrogate_norm[:n_train],
        tau_override=args.tau,
        n_override=args.n_embed,
        delay_params_file=delay_params_file,
    )
    trim = (n_embed - 1) * tau
    train_indices = list(range(trim, n_train))
    valid_test = [t for t in test_indices if t >= trim]
    print(f"tau={tau}, n={n_embed}  →  branch input: ({n_embed}, {args.proj_n}, {args.proj_n})")

    # ── 3. Load PCA basis and encode training volumes ─────────────────────────
    pca = PCAReduction(n_components=args.n_components)
    pca.load(str(artifacts_dir / "pca_basis.npz"))

    gt_volumes_dir = data_dir / "ground_truth" / "volumes"
    gt_paths = _sorted_paths(str(gt_volumes_dir), "volume_")
    if not gt_paths:
        raise FileNotFoundError(f"No GT volumes in {gt_volumes_dir}")

    print("\nEncoding training volumes with PCA ...")
    all_pca_coeffs = pca.encode_many(gt_paths[:n_train])
    full_pca_coeffs = np.zeros((args.n_timepoints, pca.n_components), dtype=np.float32)
    full_pca_coeffs[:n_train] = all_pca_coeffs

    # ── 4. Build dataset and train ────────────────────────────────────────────
    train_dataset = ProjectionDataset(
        timepoint_indices=train_indices,
        proj_small_dir=str(proj_small_dir),
        angle_idx=args.angle_idx,
        pca_coefficients=full_pca_coeffs,
        tau=tau,
        n_delay=n_embed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
    )
    print(f"Training dataset: {len(train_dataset)} samples")

    model = PODDeepONet(
        n_delay=n_embed,
        n_pca=pca.n_components,
        pca_components=pca.components,
        pca_mean=pca.mean,
        base_channels=args.base_channels,
        pool_size=args.pool_size,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"POD-DeepONet CNN branch: {n_params:,} trainable parameters")

    model, epoch_losses = train_deeponet(
        model=model,
        train_loader=train_loader,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        checkpoint_dir=str(output_dir / "checkpoints"),
        device=args.device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(output_dir / "epoch_losses.npy"), np.array(epoch_losses))

    # ── 5. Build branch inputs for test timepoints ────────────────────────────
    # Choose inference resolution: full-res or training-res projections.
    if args.inference_full_res:
        inf_dir = projections_dir
        inf_prefix = "proj_"
        print(f"\nInference at full resolution (from {inf_dir})")
    else:
        inf_dir = proj_small_dir
        inf_prefix = "proj_small_"
        print(f"\nInference at training resolution ({args.proj_n}×{args.proj_n})")

    print("Building branch inputs for test timepoints ...")
    # Determine shape from first test timepoint
    sample_stack = _load_proj_stack(inf_dir, valid_test[0], tau, n_embed,
                                    args.angle_idx, inf_prefix)
    _, H_inf, W_inf = sample_stack.shape
    branch_inputs = np.zeros(
        (args.n_timepoints, n_embed, H_inf, W_inf), dtype=np.float32
    )
    for t in valid_test:
        branch_inputs[t] = _load_proj_stack(inf_dir, t, tau, n_embed,
                                             args.angle_idx, inf_prefix)

    # ── 6. Predict on test split ──────────────────────────────────────────────
    predict_deeponet(
        model=model,
        pca=pca,
        branch_inputs=branch_inputs,
        test_indices=valid_test,
        output_dir=str(output_dir),
        ref_affine=nib.load(gt_paths[0]).affine,
    )

    print("\nDeepONet pipeline complete.")


if __name__ == "__main__":
    main()
