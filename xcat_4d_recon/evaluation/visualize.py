"""
Visualisation utilities for 4D CT reconstruction results.

Provides:
  - plot_metric_comparison : bar chart comparing methods across metrics.
  - plot_loss_curves        : training loss history for MTTDE and DeepONet.
  - save_slice_comparison   : side-by-side axial/coronal/sagittal slices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_metric_comparison(
    summary: dict[str, dict[str, float]],
    output_path: Optional[str] = None,
    metrics: Optional[list[str]] = None,
) -> None:
    """Bar chart comparing mean metric values across reconstruction methods.

    Parameters
    ----------
    summary : {method: {metric: mean_value}}
    output_path : save figure here (if None, just show).
    metrics : subset of metrics to display.
    """
    metrics = metrics or ["rmse", "ssim", "dice", "centroid_mm"]
    methods = list(summary.keys())
    n_metrics = len(metrics)
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    for ax, metric in zip(axes, metrics):
        vals = [summary[m].get(metric, float("nan")) for m in methods]
        bars = ax.bar(range(n_methods), vals, color=colors)
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.upper().replace("_", " "), fontsize=11)
        ax.set_ylabel(metric)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.suptitle("Reconstruction Method Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {output_path}")
    else:
        plt.show()
    plt.close()


def plot_loss_curves(
    loss_histories: dict[str, list[float]],
    output_path: Optional[str] = None,
    log_scale: bool = True,
) -> None:
    """Plot training loss curves for multiple methods.

    Parameters
    ----------
    loss_histories : {method_name: list_of_losses}
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, losses in loss_histories.items():
        ax.plot(losses, label=name, linewidth=1.5)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Training step / epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure → {output_path}")
    else:
        plt.show()
    plt.close()


def save_slice_comparison(
    volumes: dict[str, np.ndarray],
    output_path: Optional[str] = None,
    voxel_shape: tuple[int, int, int] = (355, 280, 115),
    clim: Optional[tuple[float, float]] = None,
) -> None:
    """Side-by-side central-slice comparison across methods.

    Parameters
    ----------
    volumes : {method_name: volume_array}  shape (x, y, z) each
    """
    n = len(volumes)
    names = list(volumes.keys())
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, name in enumerate(names):
        vol = volumes[name]
        cx, cy, cz = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2
        slices = [vol[cx, :, :], vol[:, cy, :], vol[:, :, cz]]
        titles = [f"Axial (x={cx})", f"Coronal (y={cy})", f"Sagittal (z={cz})"]
        for row, (sl, title) in enumerate(zip(slices, titles)):
            im = axes[row, col].imshow(
                sl.T,
                cmap="gray",
                origin="lower",
                vmin=clim[0] if clim else None,
                vmax=clim[1] if clim else None,
            )
            axes[row, col].set_title(f"{name}\n{title}", fontsize=9)
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.03, pad=0.02)

    plt.suptitle("Central Slices — Reconstruction Comparison", fontsize=13)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120, bbox_inches="tight")
        print(f"Saved figure → {output_path}")
    else:
        plt.show()
    plt.close()
