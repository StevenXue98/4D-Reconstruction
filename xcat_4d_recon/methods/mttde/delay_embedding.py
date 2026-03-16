"""
Time-delay embedding construction for 1D surrogate signals.

Adapted from: Measure-Theoretic-Time-Delay-Embedding-main/generate_embedding.py

Implements Takens' embedding:
  y(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (n-1)*tau)]

where tau is chosen by Mutual Information and n by False Nearest Neighbors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _mi_for_delay(signal: np.ndarray, tau_max: int = 20) -> int:
    """Estimate optimal time delay via Mutual Information (teaspoon).

    Falls back to a simple autocorrelation-based heuristic if teaspoon is
    not installed.
    """
    try:
        from teaspoon.parameter_selection.MI_delay import MI_for_delay
        tau = MI_for_delay(
            signal,
            plotting=False,
            method="basic",
            h_method="standard",
            k=2,
            ranking=True,
        )
        return int(tau)
    except ImportError:
        # Heuristic: first zero-crossing of autocorrelation
        acf = np.correlate(signal - signal.mean(), signal - signal.mean(), mode="full")
        acf = acf[len(acf) // 2 :]
        acf /= acf[0]
        crossings = np.where(acf[1:] * acf[:-1] < 0)[0]
        tau = int(crossings[0]) + 1 if len(crossings) > 0 else 5
        print(f"[delay_embedding] teaspoon not found; using autocorrelation heuristic tau={tau}")
        return min(tau, tau_max)


def _fnn_n(signal: np.ndarray, tau: int, threshold: float = 5.0, n_max: int = 10) -> int:
    """Estimate embedding dimension via False Nearest Neighbors (teaspoon).

    Falls back to n=3 (suitable for most breathing dynamics) if teaspoon
    is not installed.
    """
    try:
        from teaspoon.parameter_selection.FNN_n import FNN_n
        _, n = FNN_n(signal, tau, threshold=threshold, plotting=False, method="cao", maxDim=n_max)
        return int(n)
    except ImportError:
        print("[delay_embedding] teaspoon not found; defaulting to embedding dimension n=3")
        return 3


def build_delay_matrix(signal: np.ndarray, tau: int, n: int) -> np.ndarray:
    """Construct the delay-coordinate matrix from a 1-D signal.

    Parameters
    ----------
    signal: shape (T,)
    tau: time delay (samples)
    n: embedding dimension

    Returns
    -------
    delay_matrix: shape (T - (n-1)*tau, n)
        Row i corresponds to time t = (n-1)*tau + i.
        Column j contains x(t - j*tau)  for j = 0, 1, ..., n-1
        (i.e., column 0 is the present, last column is the oldest delay).
    """
    T = len(signal)
    trim = (n - 1) * tau
    N = T - trim
    delay_matrix = np.zeros((N, n), dtype=np.float64)
    for j in range(n):
        delay_matrix[:, j] = signal[trim - j * tau : T - j * tau if j > 0 else T]
    return delay_matrix


def compute_embedding_params(
    signal: np.ndarray,
    tau_override: Optional[int] = None,
    n_override: Optional[int] = None,
    tau_max: int = 20,
    n_max: int = 10,
    fnn_threshold: float = 5.0,
    subsample_skip: int = 1,
    delay_params_file: Optional[str] = None,
) -> tuple[int, int]:
    """Estimate (or use provided) embedding parameters and optionally cache them.

    Parameters
    ----------
    signal:
        1-D time series.
    tau_override, n_override:
        If not None, skip estimation and use these values.
    delay_params_file:
        If provided, save/load parameters from this JSON file.

    Returns
    -------
    tau, n
    """
    # Load from cache if available
    if delay_params_file and Path(delay_params_file).exists():
        with open(delay_params_file) as f:
            params = json.load(f)
        tau = params["tau"]
        n = params["n"]
        print(f"Loaded delay params from {delay_params_file}: tau={tau}, n={n}")
        return tau, n

    # Use overrides if provided
    if tau_override is not None and n_override is not None:
        tau, n = int(tau_override), int(n_override)
    else:
        subsampled = signal[::subsample_skip]
        if tau_override is None:
            tau_sub = _mi_for_delay(subsampled, tau_max=tau_max)
            tau = tau_sub * subsample_skip
        else:
            tau = int(tau_override)
            tau_sub = max(1, tau // subsample_skip)

        if n_override is None:
            n = _fnn_n(subsampled, tau_sub, threshold=fnn_threshold, n_max=n_max)
        else:
            n = int(n_override)

    print(f"Embedding parameters: tau={tau}, n={n}")

    # Save params
    if delay_params_file:
        Path(delay_params_file).parent.mkdir(parents=True, exist_ok=True)
        with open(delay_params_file, "w") as f:
            json.dump({"tau": tau, "n": n}, f, indent=2)
        print(f"Saved delay params → {delay_params_file}")

    return tau, n
