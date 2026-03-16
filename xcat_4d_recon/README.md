# 4D XCAT Reconstruction

Reconstruction of 4D respiratory-gated CT volumes from the XCAT phantom dataset,
comparing three methods:

| Method | Algorithm | Reference |
|---|---|---|
| **SuPReMo** (baseline) | B-spline motion model + surrogate signals | Huang et al., MICCAI 2024 |
| **MTTDE** | Measure-Theoretic Time-Delay Embedding (Takens' theorem + Wasserstein loss) | Measure-Theoretic paper |
| **DeepONet** | POD-DeepONet (branch net → PCA coefficients) | Lu et al. (2022) |

---

## Project Structure

```
xcat_4d_recon/
├── configs/              # YAML configuration files
│   ├── base.yaml         # Shared: data paths, voxel shape, train/test split
│   ├── supremo.yaml      # SuPReMo binary paths and variant flags
│   ├── mttde.yaml        # MTTDE hyperparameters
│   └── deeponet.yaml     # DeepONet architecture and training settings
│
├── preprocessing/        # Data preparation and shared PCA reduction
│   ├── prepare_data.py   # Extract CT slabs, sort phases, write file lists
│   ├── generate_surrogates.py   # RPM+grad and phase-derived surrogate signals
│   └── pca_reduction.py  # IncrementalPCA for 11M-voxel volume compression
│
├── methods/
│   ├── mttde/            # Measure-Theoretic Time-Delay Embedding
│   │   ├── surrogate_extraction.py   # 1D signal from CT slabs (no GT leakage)
│   │   ├── delay_embedding.py        # Takens' delay matrix + MI/FNN param selection
│   │   ├── patching.py               # Constrained k-means patch generation
│   │   ├── network.py                # 4-layer tanh MLP (delay_dim → n_pca)
│   │   ├── trainer.py                # Wasserstein/energy loss training
│   │   └── predictor.py              # Inference → NIfTI volumes
│   ├── deeponet/         # POD-DeepONet
│   │   ├── dataset.py                # CT4DDataset (delay vector or slab sub-sample)
│   │   ├── pod_deeponet.py           # Branch net + fixed PCA trunk
│   │   ├── trainer.py                # MSE-in-PCA-space training loop
│   │   └── predictor.py              # Inference → NIfTI volumes
│   └── supremo/          # SuPReMo baseline wrapper
│       ├── variants.py               # Flag sets for 3 variants
│       └── runner.py                 # subprocess wrapper
│
├── evaluation/
│   ├── metrics.py        # RMSE, SSIM, Dice, centroid displacement
│   ├── benchmark.py      # Multi-method evaluation → metrics.csv
│   └── visualize.py      # Metric comparison plots, loss curves, slice views
│
├── experiments/          # Entry-point scripts (run in order)
│   ├── prepare_data.py   # Step 1: extract slabs, generate surrogates
│   ├── fit_pca.py        # Step 2: fit shared PCA basis
│   ├── run_mttde.py      # Step 3a: train + predict MTTDE
│   ├── run_deeponet.py   # Step 3b: train + predict DeepONet
│   ├── run_baseline.py   # Step 3c: run SuPReMo (Linux only)
│   └── run_benchmark.py  # Step 4: evaluate all methods
│
├── tests/                # pytest unit tests
├── artifacts/            # Saved PCA bases, delay params, normalisation stats
├── outputs/              # Per-method reconstructed volumes and metrics
└── data/                 # Input data (not committed)
    ├── ground_truth/
    │   ├── volumes/      # volume_0.nii.gz … volume_181.nii.gz
    │   └── tumor_masks/  # mask_0.nii.gz … mask_181.nii.gz
    ├── rpm_signal.txt
    ├── timeIndicesPerSliceAndPhase.txt
    └── ref_empty_image.nii.gz
```

---

## Setup

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate xcat4d

# or pip
pip install -r requirements.txt
```

> **GPU**: install PyTorch with CUDA support:
> `pip install torch --index-url https://download.pytorch.org/whl/cu121`

---

## Data

Place the XCAT phantom data in `data/ground_truth/`:

```
data/ground_truth/volumes/volume_0.nii.gz  …  volume_181.nii.gz
data/ground_truth/tumor_masks/mask_0.nii.gz  …  mask_181.nii.gz
data/rpm_signal.txt
data/timeIndicesPerSliceAndPhase.txt
data/ref_empty_image.nii.gz
```

Dataset properties: **182 timepoints**, **355 × 280 × 115 voxels**, 1 × 1 × 3 mm spacing.
Train/test split: first **146 timepoints** (80%) train, last **36 timepoints** (20%) test.

---

## Running the Pipeline

Run scripts from the project root (`xcat_4d_recon/`):

```bash
# Step 1 – Prepare data (extract slabs, generate surrogate signals)
python experiments/prepare_data.py

# Step 2 – Fit shared PCA basis (run once; used by MTTDE and DeepONet)
python experiments/fit_pca.py --n_components 64

# Step 3a – Train and evaluate MTTDE
python experiments/run_mttde.py --n_iterations 50000

# Step 3b – Train and evaluate DeepONet
python experiments/run_deeponet.py --n_epochs 500

# Step 3c – Run SuPReMo baseline (Linux only; see note below)
python experiments/run_baseline.py

# Step 4 – Benchmark all methods
python experiments/run_benchmark.py
```

Results are written to `outputs/benchmark/metrics.csv` and `outputs/benchmark/figures/`.

---

## Platform Note: SuPReMo (Linux only)

The `runSupremo` and `animate` binaries are compiled Linux ELF executables.
On macOS, run them inside Docker:

```bash
docker run --rm \
  -v "$(pwd)":/work \
  -w /work \
  ubuntu:22.04 \
  /work/../Reference\ codebases/4DCT-irregular-motion-main/runSupremo [flags]
```

---

## Running Tests

```bash
cd xcat_4d_recon
pytest tests/ -v
```

Tests cover: PCA round-trip, delay-embedding shapes, evaluation metrics, DeepONet
architecture dimensions, and SuPReMo command construction.

---

## Method Details

### MTTDE — Measure-Theoretic Time-Delay Embedding

1. Extract a 1-D respiratory surrogate from acquired CT slabs (projection onto PCA
   component 0, or mean HU in a lung ROI).  This avoids any leakage from GT volumes
   at test time.
2. Estimate time-delay `tau` (mutual information) and embedding dimension `n`
   (false nearest neighbours) from the training segment of the surrogate.
3. Build delay-coordinate matrix: each row is `[x(t), x(t-tau), ..., x(t-(n-1)*tau)]`.
4. Encode training GT volumes to PCA coefficients (shared 64-component basis).
5. Cluster delay coordinates into balanced patches (constrained k-means, 20 clusters).
6. Train a 4-layer tanh MLP (100 hidden units) with the **energy distance** (Wasserstein
   proxy) between the *distribution* of predicted PCA coefficients and the *distribution*
   of GT PCA coefficients within each patch.
7. At test time: build delay vector from acquired surrogate → forward pass → PCA decode.

### DeepONet — POD-DeepONet

1. Use the same PCA basis as MTTDE (64 modes, fixed trunk — the POD basis).
2. Branch network (3 × 256 ReLU layers) takes the same delay vector as input, predicts
   PCA coefficients directly.
3. Train with MSE loss in PCA coefficient space (equivalent to weighted voxel-space MSE).
4. Provides a direct comparison: same branch input, same output space as MTTDE, but
   different loss (pointwise MSE vs. measure-theoretic energy distance).

### SuPReMo Baseline

Three variants of the B-spline motion model from the 4DCT reference paper:
- **Surrogate-driven**: direct least-squares fit to RPM + gradient signals.
- **Surrogate-free**: gradient-based optimisation from phase-derived sinusoidal init.
- **Surrogate-optimized**: gradient-based refinement of the clinical RPM surrogates.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root mean squared error in HU between GT and predicted volumes |
| **SSIM** | Structural similarity index (perceptual quality) |
| **Dice** | Volumetric tumour mask overlap |
| **Centroid displacement** | Euclidean distance between tumour centroids in mm |
