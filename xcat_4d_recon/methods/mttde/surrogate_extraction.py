"""
Extract a 1D surrogate respiratory signal from 2D projection images.

The surrogate is the observable used to build the time-delay embedding.
It is extracted from 2D projection images (one per timepoint, for a chosen
camera angle) so it is available from any dataset where cameras produce
2D projections — including simulated DRRs from CT phantoms and real fusion
reactor camera data.

Extraction strategy
-------------------
projection_mean : mean pixel intensity of the 2D projection image.
    Tracks respiratory (or plasma) motion because bulk displacement of the
    emitting/attenuating volume changes the total integrated intensity seen
    by the camera.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm


def extract_surrogate(projection_paths: list[str]) -> np.ndarray:
    """Extract a 1-D surrogate signal from 2D projection images.

    Each projection is a float32 .npy file containing one 2D image from a
    single camera angle.  The surrogate value at each timepoint is the mean
    pixel intensity of that image.

    Parameters
    ----------
    projection_paths:
        Ordered list of .npy projection file paths, one per timepoint.
        Typically produced by ``preprocessing.generate_projections`` and
        selected for a single angle via
        ``generate_projections.load_projection_sequence``.

    Returns
    -------
    signal : shape (n_timepoints,)  float64
    """
    signal = []
    for path in tqdm(projection_paths, desc="Extracting surrogate", leave=False):
        proj = np.load(path).astype(np.float32)
        signal.append(float(proj.mean()))
    return np.array(signal, dtype=np.float64)
