"""Tests for time-delay embedding construction."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods.mttde.delay_embedding import build_delay_matrix, compute_embedding_params


def test_delay_matrix_shape():
    signal = np.sin(np.linspace(0, 10 * np.pi, 200))
    tau, n = 3, 4
    D = build_delay_matrix(signal, tau, n)
    expected_rows = len(signal) - (n - 1) * tau
    assert D.shape == (expected_rows, n)


def test_delay_matrix_values():
    signal = np.arange(20, dtype=float)
    tau, n = 2, 3
    D = build_delay_matrix(signal, tau, n)
    # Row 0 corresponds to t = (n-1)*tau = 4
    # Column 0: x[4] = 4, column 1: x[2] = 2, column 2: x[0] = 0
    assert D[0, 0] == pytest.approx(4.0)
    assert D[0, 1] == pytest.approx(2.0)
    assert D[0, 2] == pytest.approx(0.0)


def test_delay_matrix_tau1_n1():
    signal = np.random.randn(50)
    D = build_delay_matrix(signal, tau=1, n=1)
    assert D.shape == (50, 1)
    np.testing.assert_array_equal(D[:, 0], signal)


def test_compute_params_override(tmp_path):
    signal = np.sin(np.linspace(0, 4 * np.pi, 100))
    params_file = str(tmp_path / "params.json")
    tau, n = compute_embedding_params(
        signal, tau_override=5, n_override=3, delay_params_file=params_file
    )
    assert tau == 5
    assert n == 3

    # Should load from cache on second call
    tau2, n2 = compute_embedding_params(signal, delay_params_file=params_file)
    assert tau2 == 5
    assert n2 == 3
