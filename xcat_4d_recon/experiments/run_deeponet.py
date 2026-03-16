"""
Step 3b: Train and evaluate the POD-DeepONet reconstruction method.

Full pipeline:
  1. Load PCA basis.
  2. Prepare branch inputs (delay vectors matching MTTDE for fair comparison,
     or CT slab sub-samples).
  3. Encode training volumes to PCA coefficients.
  4. Train POD-DeepONet branch network (MSE in PCA coefficient space).
  5. Predict on test timepoints and save NIfTI volumes.

Usage:
    python experiments/run_deeponet.py
    python experiments/run_deeponet.py --branch_method slab_subsample --n_epochs 500
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
from methods.mttde.surrogate_extraction import extract_surrogate
from methods.mttde.delay_embedding import compute_embedding_params, build_delay_matrix
from methods.deeponet.dataset import CT4DDataset
from methods.deeponet.pod_deeponet import PODDeepONet
from methods.deeponet.trainer import train_deeponet
from methods.deeponet.predictor import predict_deeponet


def _sorted_paths(directory: str, prefix: str) -> list[str]:
    paths = glob(str(Path(directory) / f"{prefix}*.nii.gz"))
    paths.sort(key=lambda p: int(Path(p).stem.split("_")[-1].split(".")[0]))
    return paths


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
    # Branch input
    parser.add_argument("--branch_method", default="delay_vector",
                        choices=["delay_vector", "slab_subsample"])
    parser.add_argument("--slab_subsample_m", type=int, default=512)
    parser.add_argument("--surrogate_method", default="pca_mode",
                        choices=["pca_mode", "mean_hu"])
    # Network
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--activation", default="relu", choices=["relu", "tanh"])
    # Training
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
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
    n_train = int(args.n_timepoints * args.train_frac)
    test_indices = list(range(n_train, args.n_timepoints))
    print(f"Train: 0..{n_train-1}  Test: {n_train}..{args.n_timepoints-1}")

    # ── 1. Load PCA ───────────────────────────────────────────────────────────
    basis_file = str(artifacts_dir / "pca_basis.npz")
    pca = PCAReduction(n_components=args.n_components)
    pca.load(basis_file)

    # ── 2. Encode ALL training volumes to PCA coefficients ────────────────────
    gt_volumes_dir = data_dir / "ground_truth" / "volumes"
    gt_paths = _sorted_paths(str(gt_volumes_dir), "volume_")
    if not gt_paths:
        raise FileNotFoundError(f"No GT volumes in {gt_volumes_dir}")

    print("\nEncoding training volumes with PCA ...")
    all_pca_coeffs = pca.encode_many(gt_paths[:n_train])  # (n_train, n_pca)
    # Pad to full n_timepoints length (test entries unused but keeps indexing simple)
    full_pca_coeffs = np.zeros((args.n_timepoints, args.n_components), dtype=np.float32)
    full_pca_coeffs[:n_train] = all_pca_coeffs

    # ── 3. Prepare branch inputs ──────────────────────────────────────────────
    delay_matrix = None
    delay_offset = 0
    branch_input_dim = args.n_components  # placeholder, updated below

    if args.branch_method == "delay_vector":
        # Re-use MTTDE embedding (fair comparison)
        slabs_dir = data_dir / "unsort_ct_slabs"
        all_slab_paths = _sorted_paths(str(slabs_dir), "slab_")
        print("\nExtracting 1D surrogate signal ...")
        surrogate = extract_surrogate(
            slab_paths=all_slab_paths,
            method=args.surrogate_method,
            pca_mean=pca.mean if args.surrogate_method == "pca_mode" else None,
            pca_component0=pca.components[0] if args.surrogate_method == "pca_mode" else None,
            volume_shape=pca._volume_shape,
        )
        scale = float(np.max(np.abs(surrogate)))
        surrogate_norm = surrogate / scale if scale > 0 else surrogate

        delay_params_file = str(artifacts_dir / "delay_params.json")
        tau, n_embed = compute_embedding_params(
            signal=surrogate_norm[:n_train],
            delay_params_file=delay_params_file,  # load cached if exists
        )
        delay_matrix = build_delay_matrix(surrogate_norm, tau, n_embed)
        delay_offset = (n_embed - 1) * tau
        branch_input_dim = n_embed
        print(f"Branch input: delay vector (tau={tau}, n={n_embed}, dim={branch_input_dim})")

    elif args.branch_method == "slab_subsample":
        branch_input_dim = args.slab_subsample_m
        print(f"Branch input: slab sub-sample (m={branch_input_dim})")

    # ── 4. Build dataset and DataLoader ──────────────────────────────────────
    # Training indices must be in the valid range of the delay matrix
    if args.branch_method == "delay_vector":
        train_indices = list(range(delay_offset, n_train))
    else:
        train_indices = list(range(n_train))

    train_dataset = CT4DDataset(
        timepoint_indices=train_indices,
        pca_coefficients=full_pca_coeffs,
        branch_method=args.branch_method,
        delay_matrix=delay_matrix,
        delay_offset=delay_offset,
        slab_dir=str(data_dir / "unsort_ct_slabs"),
        slab_subsample_m=args.slab_subsample_m,
        subsample_seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=0,
    )
    print(f"Training dataset: {len(train_dataset)} samples")

    # ── 5. Build POD-DeepONet model ───────────────────────────────────────────
    model = PODDeepONet(
        branch_input_dim=branch_input_dim,
        hidden_dims=args.hidden_dims,
        n_pca=args.n_components,
        pca_components=pca.components,
        pca_mean=pca.mean,
        activation=args.activation,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"POD-DeepONet branch net: {n_params:,} trainable parameters")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    checkpoint_dir = str(output_dir / "checkpoints")
    model, epoch_losses = train_deeponet(
        model=model,
        train_loader=train_loader,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        checkpoint_dir=checkpoint_dir,
        device=args.device,
    )
    np.save(str(output_dir / "epoch_losses.npy"), np.array(epoch_losses))

    # ── 7. Predict on test split ──────────────────────────────────────────────
    ref_affine = nib.load(gt_paths[0]).affine

    if args.branch_method == "delay_vector":
        # Build branch inputs for all timepoints from delay matrix
        # test_indices in delay_matrix alignment: t - delay_offset
        valid_test = [t for t in test_indices if (t - delay_offset) < len(delay_matrix)]
        branch_inputs = delay_matrix.astype(np.float32)  # (T - trim, n_embed)

        # predict_deeponet expects branch_inputs indexed by t directly; adjust
        # by building a padded array indexed by absolute timepoint
        padded = np.zeros((args.n_timepoints, n_embed), dtype=np.float32)
        for t in range(delay_offset, args.n_timepoints):
            if t - delay_offset < len(delay_matrix):
                padded[t] = delay_matrix[t - delay_offset]

        predict_deeponet(
            model=model,
            pca=pca,
            branch_inputs=padded,
            test_indices=valid_test,
            output_dir=str(output_dir),
            ref_affine=ref_affine,
        )
    else:
        # slab_subsample: build branch inputs on the fly in predict_deeponet
        # Use a simple loop via dataset
        _predict_slab_subsample(
            model, pca, train_dataset, test_indices, output_dir, ref_affine, args
        )

    print("\nDeepONet pipeline complete.")


def _predict_slab_subsample(model, pca, train_dataset, test_indices, output_dir, ref_affine, args):
    """Fallback predictor for slab_subsample branch method (builds inputs lazily)."""
    import nibabel as nib
    vol_dir = Path(output_dir) / "estimated_volumes"
    vol_dir.mkdir(parents=True, exist_ok=True)

    from methods.deeponet.dataset import CT4DDataset
    from pathlib import Path as P
    from tqdm import tqdm

    slabs_dir = Path(args.data_dir) / "unsort_ct_slabs"

    model.eval()
    for t in tqdm(test_indices, desc="DeepONet predict"):
        test_ds = CT4DDataset(
            timepoint_indices=[t],
            pca_coefficients=np.zeros((args.n_timepoints, args.n_components)),
            branch_method="slab_subsample",
            slab_dir=str(slabs_dir),
            slab_subsample_m=args.slab_subsample_m,
            subsample_seed=args.seed,
        )
        branch, _ = test_ds[0]
        with torch.no_grad():
            coeffs = model.forward_coefficients(branch.unsqueeze(0)).squeeze(0).numpy()
        volume = pca.decode(coeffs)
        nib.nifti1.save(nib.Nifti1Image(volume, ref_affine), str(vol_dir / f"volume_{t}.nii.gz"))


if __name__ == "__main__":
    main()
