"""
Generate surrogate respiratory motion signals.

Adapted from: 4DCT-irregular-motion-main/generate_surrogate_signals.py (Huang et al., MICCAI 2024)

Two surrogate types are produced:
  1. RPM + gradient (z-scored)  → used by surrogate-driven and surrogate-optimized methods.
  2. Phase-derived sinusoids    → initialisation for surrogate-free method.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import zscore


def generate_surrogates(data_dir: str) -> None:
    """Create both surrogate signal files inside *data_dir*.

    Parameters
    ----------
    data_dir:
        Root data directory.  Must contain ``rpm_signal.txt`` and
        ``timeIndicesPerSliceAndPhase.txt``.
    """
    data_dir = Path(data_dir)

    rpm = np.loadtxt(data_dir / "rpm_signal.txt")
    rpm_grad = np.gradient(rpm)
    surrogate_rpm_grad = np.stack([zscore(rpm), zscore(rpm_grad)], axis=-1)
    rpm_grad_path = data_dir / "surrogate_rpm_grad.txt"
    np.savetxt(rpm_grad_path, surrogate_rpm_grad, fmt="%f")
    print(f"Written: {rpm_grad_path}  shape={surrogate_rpm_grad.shape}")

    time_indices = np.loadtxt(
        data_dir / "timeIndicesPerSliceAndPhase.txt", dtype=int
    )
    n_timepoints = rpm.shape[0]
    phases = np.array(
        [np.where(time_indices == i)[1][0] for i in range(n_timepoints)]
    )
    n_phases = time_indices.shape[1]
    surr_phase = np.stack(
        [np.sin(2 * np.pi * phases / n_phases), np.cos(2 * np.pi * phases / n_phases)],
        axis=-1,
    )
    phase_path = data_dir / "surrogate_phase_derived.txt"
    np.savetxt(phase_path, surr_phase, fmt="%f")
    print(f"Written: {phase_path}  shape={surr_phase.shape}")
