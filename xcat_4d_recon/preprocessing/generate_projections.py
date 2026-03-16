"""
Generate simulated 2D X-ray projections from 3D CT volumes (parallel-beam DRR).

Simulates cameras placed at fixed angles around the volume.  Each camera produces
a 2D projection by rotating the volume to the desired viewing angle and then
integrating (summing) HU values along the depth axis — equivalent to a parallel-beam
digitally reconstructed radiograph (DRR).

This mimics the observation model in fusion reactor diagnostics, where cameras are
placed at fixed positions around the torus and each records a 2D line-of-sight
integrated image of the plasma.

Geometry
--------
- Rotation is in the x-y plane (around the z / head-foot axis).
- After rotating by angle θ, the volume is summed along the y-axis.
- The resulting projection has shape (x, z) = (355, 115) for the XCAT phantom.
- Angles are evenly spaced in [angle_start, angle_start + angle_range) degrees.

Output
------
Projections are saved as float32 .npy files:
    <output_dir>/proj_<vol_idx:03d>_angle_<angle_idx:02d>.npy

A metadata file is also written:
    <output_dir>/projection_meta.npz
        angles_deg  : (n_angles,) array of viewing angles
        n_volumes   : int
        volume_shape: (x, y, z)
        proj_shape  : (x, z)
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import rotate
from tqdm import tqdm


def generate_projections(
    volume_paths: list[str],
    output_dir: str,
    n_angles: int = 6,
    angle_start: float = 0.0,
    angle_range: float = 180.0,
) -> list[list[str]]:
    """Generate parallel-beam 2D projections from a list of 3D NIfTI volumes.

    For each volume, ``n_angles`` projections are produced by rotating the volume
    in the x-y plane and summing along the y-axis.

    Parameters
    ----------
    volume_paths:
        Ordered list of .nii.gz paths (e.g. ground_truth volumes).
    output_dir:
        Directory where projection .npy files are written.
    n_angles:
        Number of camera positions.  Angles are evenly spaced in
        [angle_start, angle_start + angle_range).
    angle_start:
        Starting angle in degrees (default 0°).
    angle_range:
        Total angular span in degrees (default 180°).  Using 180° with
        parallel-beam geometry gives non-redundant views; use 360° to include
        all directions.

    Returns
    -------
    projection_paths : list[list[str]]
        Shape (n_volumes, n_angles).
        ``projection_paths[i][j]`` is the path to the projection of volume ``i``
        at angle index ``j``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    angles = np.linspace(angle_start, angle_start + angle_range, n_angles, endpoint=False)

    all_paths: list[list[str]] = []
    volume_shape = None
    proj_shape = None

    for vol_idx, vol_path in enumerate(tqdm(volume_paths, desc="Generating projections")):
        vol = nib.load(vol_path).get_fdata().astype(np.float32)

        if volume_shape is None:
            volume_shape = vol.shape

        vol_proj_paths: list[str] = []
        for angle_idx, angle in enumerate(angles):
            # Rotate in the x-y plane (axes 0 and 1) around the z-axis
            rotated = rotate(
                vol,
                angle=float(angle),
                axes=(0, 1),
                reshape=False,
                order=1,          # bilinear — fast, sufficient for surrogate extraction
                mode="constant",
                cval=0.0,
            )
            # Integrate (sum) along y-axis → projection shape is (x, z)
            projection = rotated.sum(axis=1).astype(np.float32)

            if proj_shape is None:
                proj_shape = projection.shape

            out_path = output_dir / f"proj_{vol_idx:03d}_angle_{angle_idx:02d}.npy"
            np.save(str(out_path), projection)
            vol_proj_paths.append(str(out_path))

        all_paths.append(vol_proj_paths)

    # Save metadata so downstream code doesn't need to recompute angles
    np.savez(
        str(output_dir / "projection_meta.npz"),
        angles_deg=angles,
        n_volumes=np.array(len(volume_paths)),
        volume_shape=np.array(volume_shape) if volume_shape else np.array([]),
        proj_shape=np.array(proj_shape) if proj_shape else np.array([]),
    )
    print(
        f"Projections written to {output_dir}\n"
        f"  Volumes   : {len(volume_paths)}\n"
        f"  Angles    : {n_angles}  {np.round(angles, 1).tolist()} deg\n"
        f"  Proj shape: {proj_shape}"
    )
    return all_paths


def load_projection(path: str) -> np.ndarray:
    """Load a single projection .npy file. Returns float32 array (x, z)."""
    return np.load(path).astype(np.float32)


def load_projection_sequence(
    projection_paths: list[list[str]],
    angle_idx: int = 0,
) -> np.ndarray:
    """Load the projection sequence for one camera angle across all timepoints.

    Parameters
    ----------
    projection_paths:
        Nested list returned by :func:`generate_projections`, shape (T, n_angles).
    angle_idx:
        Which camera angle to load.

    Returns
    -------
    sequence : shape (T, x, z)  float32
    """
    return np.stack(
        [load_projection(paths[angle_idx]) for paths in projection_paths],
        axis=0,
    )


def downsample_projections(
    input_dir: str,
    output_dir: str,
    target_h: int = 15,
    target_w: int = 15,
) -> None:
    """Downsample all projection images to a smaller spatial resolution.

    For each ``proj_<vol>_angle_<a>.npy`` in *input_dir*, applies Gaussian
    smoothing (sigma proportional to the downsampling ratio — prevents aliasing
    and suppresses noise) then resizes to ``(target_h, target_w)`` using
    bilinear interpolation.  Output files are named
    ``proj_small_<vol>_angle_<a>.npy`` and written to *output_dir*.

    The Gaussian sigma is chosen as ``scale_factor / 2`` per axis, which
    corresponds to a Nyquist-frequency pre-filter: it removes spatial
    frequencies that cannot be represented at the target resolution while
    preserving lower-frequency structure (i.e., the bulk motion signal).

    Parameters
    ----------
    input_dir:
        Directory containing full-resolution ``proj_*.npy`` files.
    output_dir:
        Directory where downsampled ``proj_small_*.npy`` files are written.
    target_h:
        Output height (first spatial axis, x in XCAT geometry).
    target_w:
        Output width (second spatial axis, z in XCAT geometry).
    """
    from scipy.ndimage import gaussian_filter, zoom

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proj_files = sorted(input_dir.glob("proj_[0-9]*.npy"))
    if not proj_files:
        raise FileNotFoundError(f"No projection files found in {input_dir}")

    # Infer original shape from first file
    sample = np.load(str(proj_files[0]))
    orig_h, orig_w = sample.shape
    sigma_h = (orig_h / target_h) / 2.0
    sigma_w = (orig_w / target_w) / 2.0
    zoom_h = target_h / orig_h
    zoom_w = target_w / orig_w

    print(
        f"Downsampling {len(proj_files)} projections  "
        f"{orig_h}×{orig_w} → {target_h}×{target_w}  "
        f"σ=({sigma_h:.2f}, {sigma_w:.2f})"
    )

    for fpath in tqdm(proj_files, desc="Downsampling projections"):
        proj = np.load(str(fpath)).astype(np.float32)
        smoothed = gaussian_filter(proj, sigma=[sigma_h, sigma_w])
        small = zoom(smoothed, [zoom_h, zoom_w], order=1).astype(np.float32)
        out_name = "proj_small_" + fpath.name[len("proj_"):]
        np.save(str(output_dir / out_name), small)

    # Save metadata
    np.savez(
        str(output_dir / "proj_small_meta.npz"),
        target_shape=np.array([target_h, target_w]),
        orig_shape=np.array([orig_h, orig_w]),
    )
    print(f"Downsampled projections written to {output_dir}")
