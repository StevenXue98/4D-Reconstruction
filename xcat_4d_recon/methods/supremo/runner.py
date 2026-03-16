"""
Python wrapper around the SuPReMo C++ binary.

SuPReMo fits a B-spline-based respiratory motion model from unsorted 4DCT slabs
and surrogate signals.  The companion `animate` binary then samples the motion
model at each timepoint to produce reconstructed volumes.

Platform note:
  runSupremo is a Linux ELF binary.  On macOS, run inside a Docker container or
  WSL.  This wrapper will raise a clear error if the binary is not executable.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from .variants import SupremoVariantConfig


def _check_binary(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"SuPReMo binary not found: {path}\n"
            "Make sure the 4DCT-irregular-motion-main reference codebase is present "
            "and that the binary is compiled for the current platform."
        )
    if not os.access(path, os.X_OK):
        raise PermissionError(
            f"SuPReMo binary is not executable: {path}\n"
            "Run: chmod +x {path}"
        )
    if platform.system() == "Darwin":
        import warnings
        warnings.warn(
            "Detected macOS.  runSupremo is a Linux ELF binary and will not run "
            "natively.  Consider using Docker (e.g., docker run --rm -v $(pwd):/work "
            "ubuntu:22.04 /work/runSupremo ...) or WSL.",
            RuntimeWarning,
            stacklevel=3,
        )


def build_supremo_command(cfg: SupremoVariantConfig) -> list[str]:
    """Construct the runSupremo command-line invocation.

    Parameters
    ----------
    cfg: SupremoVariantConfig

    Returns
    -------
    list of str – command + arguments
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        cfg.binary,
        "-dynImgs", cfg.dynamic_image_files,
        "-surr", cfg.surrogate_file,
        "-refImg", cfg.ref_image,
        "-out", str(out_dir / "motion_model"),
        "-simMeasure", cfg.similarity_measure,
        "-transType", cfg.transformation_type,
        "-bSplineSpacing", str(cfg.b_spline_spacing),
        "-optimiserType", str(cfg.optimizer_type),
        "-nThreads", str(cfg.n_threads),
    ]
    return cmd


def build_animate_command(cfg: SupremoVariantConfig, dynamic_image_files: str) -> list[str]:
    """Construct the animate command to generate per-timepoint volumes."""
    out_dir = Path(cfg.output_dir)
    vol_dir = out_dir / "estimated_volumes"
    mask_dir = out_dir / "estimated_tumormasks"
    vol_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        cfg.animate_binary,
        "-motionModel", str(out_dir / "motion_model.nii.gz"),
        "-surr", cfg.surrogate_file,
        "-dynImgs", dynamic_image_files,
        "-out", str(vol_dir / "volume"),
        "-outMask", str(mask_dir / "mask"),
    ]
    return cmd


def run_supremo(cfg: SupremoVariantConfig, dry_run: bool = False) -> None:
    """Execute the full SuPReMo pipeline for one variant.

    Parameters
    ----------
    cfg:
        Variant-specific configuration.
    dry_run:
        If True, print commands but do not execute them.
    """
    _check_binary(cfg.binary)

    supremo_cmd = build_supremo_command(cfg)
    animate_cmd = build_animate_command(cfg, cfg.dynamic_image_files)

    print(f"\n[{cfg.variant_name}] Running SuPReMo motion model fitting ...")
    print("  CMD:", " ".join(supremo_cmd))
    if not dry_run:
        result = subprocess.run(supremo_cmd, check=True, capture_output=False)

    print(f"\n[{cfg.variant_name}] Animating motion model to reconstruct volumes ...")
    print("  CMD:", " ".join(animate_cmd))
    if not dry_run:
        result = subprocess.run(animate_cmd, check=True, capture_output=False)

    print(f"[{cfg.variant_name}] Done.  Outputs in: {cfg.output_dir}")
