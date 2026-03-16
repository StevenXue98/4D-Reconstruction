"""
Flag definitions for the three SuPReMo motion-model variants.

Reference: 4DCT-irregular-motion-main README and runSupremo usage.

The three variants differ in:
  - optimiser_type: 0 (surrogate-driven) vs. 2 (gradient-based)
  - surrogate_file: RPM+gradient vs. phase-derived sinusoids
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SupremoVariantConfig:
    """Runtime parameters for one SuPReMo execution."""
    variant_name: str
    optimizer_type: int           # 0 = surrogate-driven, 2 = gradient-based
    surrogate_file: str           # path to N_timepoints × N_surrogates txt file
    output_dir: str
    dynamic_image_files: str      # path to dynamic_image_files.txt
    ref_image: str                # reference state NIfTI
    binary: str                   # path to runSupremo executable
    animate_binary: str           # path to animate executable
    n_threads: int = 4
    similarity_measure: str = "SSD"
    b_spline_spacing: int = 20
    transformation_type: str = "B-Spline"


def get_variant_configs(
    data_dir: str,
    output_dir: str,
    binary: str,
    animate_binary: str,
    n_threads: int = 4,
    train_only: bool = False,
) -> dict[str, SupremoVariantConfig]:
    """Build variant configs for all three SuPReMo methods.

    Parameters
    ----------
    data_dir:
        Root data directory.
    output_dir:
        Root output directory (subdirectories are created per variant).
    binary, animate_binary:
        Paths to SuPReMo executables.
    train_only:
        If True, use dynamic_image_files_train.txt (fit on train split only).
    """
    data_dir = Path(data_dir)
    img_file = (
        str(data_dir / "dynamic_image_files_train.txt")
        if train_only
        else str(data_dir / "dynamic_image_files.txt")
    )
    ref_image = str(data_dir / "ref_empty_image.nii.gz")

    shared = dict(
        dynamic_image_files=img_file,
        ref_image=ref_image,
        binary=binary,
        animate_binary=animate_binary,
        n_threads=n_threads,
    )

    return {
        "surrogate_driven": SupremoVariantConfig(
            variant_name="surrogate_driven",
            optimizer_type=0,
            surrogate_file=str(data_dir / "surrogate_rpm_grad.txt"),
            output_dir=str(Path(output_dir) / "surr_driven"),
            **shared,
        ),
        "surrogate_free": SupremoVariantConfig(
            variant_name="surrogate_free",
            optimizer_type=2,
            surrogate_file=str(data_dir / "surrogate_phase_derived.txt"),
            output_dir=str(Path(output_dir) / "surr_free"),
            **shared,
        ),
        "surrogate_optimized": SupremoVariantConfig(
            variant_name="surrogate_optimized",
            optimizer_type=2,
            surrogate_file=str(data_dir / "surrogate_rpm_grad.txt"),
            output_dir=str(Path(output_dir) / "surr_optimized"),
            **shared,
        ),
    }
