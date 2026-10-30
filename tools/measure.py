"""Measurements on masks and cancer maps shared by the tools: outline quality, hot spots, values along a region."""

from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

# hot spots: connected region of the map >= 0.5 covering >= 0.2% of the frame and >= 50% inside the gland; its hot core =
# pixels >= max(0.7, 0.8 x the region's maximum); summarised as the circle of the hot core's area at its centroid
HOT_THRESHOLD, HOT_ABS, HOT_REL, MIN_AREA_FRAC, MAX_HOT_SPOTS = 0.5, 0.7, 0.8, 0.002, 5


def resize_mask(mask: np.ndarray, shape) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)) > 0


def resize_map(a: np.ndarray, shape) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize((shape[1], shape[0]), Image.BILINEAR), np.float32)


def mask_metrics(mask: np.ndarray) -> dict:
    """Coverage of the frame, number of components, centroid offset of the largest component from the frame centre (as a
    fraction of the frame size) and contact with the frame edge."""
    h, w = mask.shape
    if not mask.any():
        return {"coverage_frac": 0.0, "n_components": 0, "centroid_offset_frac": None, "touches_edge": False}
    props = regionprops(label(mask))
    cy, cx = max(props, key=lambda p: p.area).centroid
    return {
        "coverage_frac": round(float(mask.mean()), 4),
        "n_components": len(props),
        "centroid_offset_frac": round(float(np.hypot((cy - h / 2) / h, (cx - w / 2) / w)), 4),
        "touches_edge": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    }


def gland_is_valid(m: dict) -> bool:
    """A plausible prostate outline: at most two components, >= 5% of the frame, off the edges, roughly centred."""
    return (
        m["n_components"] <= 2
        and m["coverage_frac"] >= 0.05
        and not m["touches_edge"]
        and (m["centroid_offset_frac"] or 1.0) <= 0.30
    )


def hot_spots(heatmap: np.ndarray, gland: np.ndarray) -> list[dict]:
    """The map's suspicious regions inside the gland, largest first (at most five)."""
    out = []
    lab = label(heatmap >= HOT_THRESHOLD)
    for p in sorted(regionprops(lab), key=lambda r: -r.area):
        if p.area < MIN_AREA_FRAC * heatmap.size:
            continue
        m = lab == p.label
        inside = float((m & gland).sum() / m.sum()) if gland.any() else 0.0
        if inside < 0.5:
            continue
        peak = float(heatmap[m].max())
        hot = m & (heatmap >= max(HOT_ABS, HOT_REL * peak))
        if not hot.any():
            continue
        ys, xs = np.nonzero(hot)
        out.append(
            {
                "cx": float(xs.mean()),
                "cy": float(ys.mean()),
                "r_px": float(np.sqrt(hot.sum() / np.pi)),
                "hot_area_frac": float(hot.sum() / heatmap.size),
                "blob_area_frac": float(m.sum() / heatmap.size),
                "max_prob": peak,
                "hot_mean": float(heatmap[hot].mean()),
                "frac_inside_gland": inside,
            }
        )
    return out[:MAX_HOT_SPOTS]


def mean_in(heatmap: np.ndarray, mask: np.ndarray | None) -> float:
    """Mean map value inside a mask (the specialists' core score inside the needle track); NaN if the mask is empty."""
    if mask is None:
        return np.nan
    m = resize_mask(mask, heatmap.shape) if mask.shape != heatmap.shape else mask
    return float(heatmap[m].mean()) if m.any() else np.nan


def along_path(heatmap: np.ndarray, path: np.ndarray) -> dict:
    """The map along a needle path given as a weight map (the mean annotated track): weighted mean, and max / 90th
    percentile over the path region (weight >= 30% of its maximum). The map is resampled to the path's grid."""
    a = resize_map(heatmap, path.shape)
    region = path >= 0.3 * path.max()
    return {
        "wmean": float((a * path).sum() / path.sum()),
        "max": float(a[region].max()),
        "p90": float(np.percentile(a[region], 90)),
    }


def needle_geometry(track: np.ndarray, mm_per_px: tuple[float, float]) -> dict | None:
    """Length, angle and depth range of a needle track."""
    if not track.any():
        return None
    dy, dx = mm_per_px
    p = max(regionprops(label(track)), key=lambda r: r.area)
    vy, vx = np.cos(p.orientation) * p.axis_major_length * dy, np.sin(p.orientation) * p.axis_major_length * dx
    rows = np.nonzero(track.any(1))[0]
    return {
        "length_mm": round(float(np.hypot(vy, vx)), 1),
        "angle_deg": round(float(np.degrees(np.arctan2(abs(vy), abs(vx)))), 1),
        "depth_mm": [round(float(rows.min() * dy), 1), round(float(rows.max() * dy), 1)],
    }
