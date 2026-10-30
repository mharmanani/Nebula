"""The lesion on a core's frame: the hottest point of the average of both cancer maps (each smoothed with a 1 mm Gaussian)
inside the prostate outline, as a 4 mm disc. Computed on a quarter-resolution grid with each axis's own mm per pixel
(micro-ultrasound pixels are not square). Whether it is shown as the suggested lesion is decided by the composer."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from .measure import resize_map, resize_mask

RADIUS_MM, DOWNSAMPLE, SMOOTH_MM = 4.0, 4, 1.0


def find_lesion(pnf, dino, gland, mm_per_px, guide_path=None, track=None) -> dict:
    """pnf, dino: (H, W) maps; gland, track: (H, W) masks; guide_path: weight map of the usual needle path (any grid);
    mm_per_px: (down, across) at full resolution."""
    h, w = pnf.shape
    shape = (int(np.ceil(h / DOWNSAMPLE)), int(np.ceil(w / DOWNSAMPLE)))
    sy, sx = mm_per_px[0] * DOWNSAMPLE, mm_per_px[1] * DOWNSAMPLE
    smooth = lambda m: gaussian_filter(resize_map(m, shape), (SMOOTH_MM / sy, SMOOTH_MM / sx))
    p, d = smooth(pnf), smooth(dino)
    avg = (p + d) / 2
    g = resize_mask(gland, shape)
    region = g if g.any() else np.ones(shape, bool)
    cy, cx = np.unravel_index(np.where(region, avg, -1).argmax(), shape)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    disc = ((yy - cy) * sy) ** 2 + ((xx - cx) * sx) ** 2 <= RADIUS_MM**2
    out = {
        "centre_depth_mm": round(float((cy * DOWNSAMPLE + DOWNSAMPLE // 2) * mm_per_px[0]), 1),
        "radius_mm": RADIUS_MM,
        "combined_map_value": round(float(avg[cy, cx]), 3),
        "prostnfound_value": round(float(p[cy, cx]), 3),
        "dinov3_value": round(float(d[cy, cx]), 3),
        "gland_found": bool(g.any()),
    }
    if guide_path is not None:
        gp = resize_map(guide_path, shape)
        gp = gp >= 0.3 * gp.max()
        out["distance_from_guide_line_mm"] = round(float(distance_transform_edt(~gp, sampling=(sy, sx))[cy, cx]), 1)
    if track is not None and track.any():
        t = resize_mask(track, shape)
        out.update(
            {
                "needle_track_crosses_lesion": bool((disc & t).any()),
                "lesion_centre_to_track_mm": round(float(distance_transform_edt(~t, sampling=(sy, sx))[cy, cx]), 1),
            }
        )
    return out
