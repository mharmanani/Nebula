"""Label-free registration of a biopsy frame with a sweep frame (both sagittal planes at nearly the same probe roll).

The operator slides the probe to reach apex or base, so a core's frame and the nearest sweep frame are shifted along the
probe. Both images go on a common 0.5 mm grid, are blurred and standardised, and the shift (depth +-2 mm, any lateral shift
keeping >= 20 mm of overlap) with the highest normalised cross-correlation is kept.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

GRID_MM = 0.5
MIN_OVERLAP, MAX_DEPTH_SHIFT = int(20 / GRID_MM), int(2 / GRID_MM)


def to_grid(a: np.ndarray, mm_per_px: tuple[float, float], order: int = 1) -> np.ndarray:
    return zoom(np.asarray(a, np.float32), (mm_per_px[0] / GRID_MM, mm_per_px[1] / GRID_MM), order=order)


def standardise(a: np.ndarray) -> np.ndarray:
    g = gaussian_filter(a, 2.0)
    return (g - g.mean()) / (g.std() + 1e-6)


def register(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, int, int]:
    """(ncc, dy, dx) on the grid such that fixed[r, c] ~ moving[r - dy, c - dx]; both already standardised on the grid."""
    best = (-2.0, 0, 0)
    for dy in range(-MAX_DEPTH_SHIFT, MAX_DEPTH_SHIFT + 1):
        for dx in range(-(moving.shape[1] - MIN_OVERLAP), fixed.shape[1] - MIN_OVERLAP + 1):
            r0, r1 = max(0, dy), min(fixed.shape[0], moving.shape[0] + dy)
            c0, c1 = max(0, dx), min(fixed.shape[1], moving.shape[1] + dx)
            if c1 - c0 < MIN_OVERLAP or r1 - r0 < fixed.shape[0] // 2:
                continue
            f = fixed[r0:r1, c0:c1]
            m = moving[r0 - dy : r1 - dy, c0 - dx : c1 - dx]
            fz, mz = f - f.mean(), m - m.mean()
            v = float((fz * mz).sum() / (np.sqrt((fz * fz).sum() * (mz * mz).sum()) + 1e-9))
            if v > best[0]:
                best = (v, dy, dx)
    return best


def register_frames(core_frame, core_mm, sweep_frame, sweep_mm) -> tuple[float, int, int]:
    """Register a core's frame (fixed) with a sweep frame (moving); shifts in grid pixels of GRID_MM."""
    return register(standardise(to_grid(core_frame, core_mm)), standardise(to_grid(sweep_frame, sweep_mm)))
