"""Runs the specialists on one case, on demand, and keeps their outputs so every tool of an episode reads the same maps.

Per core (its fire frame): the gland outline (MicroSegNet), the needle track (NeedleNet: as in the paper, no annotation is
used at inference), both cancer maps, and the core scores (mean map value inside the track). Per sweep frame: the gland outline, both maps and their summaries inside the gland, and the hot spots.
Per study: Compass. Nothing here reads a label.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd

from .measure import gland_is_valid, hot_spots, mask_metrics, mean_in, resize_map
from .registration import GRID_MM, register_frames

MIN_SWEEP_GLAND = 0.05  # sweep frames with less gland than this are not scored
VOLUME_PROBE_RADIUS_MM, VOLUME_BIN_DEG = 10.0, 2.0


class CaseAnalysis:
    def __init__(self, case, specialists: dict):
        self.case, self.sp = case, specialists
        self._core = {}

    # ── cores ────────────────────────────────────────────────────────────────────────────────────────────────────────
    def core(self, core) -> dict:
        """Frame, gland, track and both maps on a core's fire frame (the live frame while it is aimed)."""
        if core.core_id not in self._core:
            frame = core.frame()
            gland = self.sp["microsegnet"].segment(frame)
            self._core[core.core_id] = {
                "frame": frame,
                "gland": gland,
                "gland_metrics": mask_metrics(gland),
                "pnf": self.sp["prostnfound"].heatmap(frame, self.case.age, self.case.psa),
                "dino": self.sp["dinov3"].heatmap(frame),
                "mm_per_px": core.mm_per_px(frame.shape),
            }
        return self._core[core.core_id]

    def needle_track(self, core) -> np.ndarray:
        c = self.core(core)
        if "track" not in c:
            c["track"] = self.sp["needlenet"].segment(c["frame"])
        return c["track"]

    @cached_property
    def core_scores(self) -> pd.DataFrame:
        """Per core: ProstNFound+ and DINOv3 mean inside the needle track, and Compass's per-core logit difference."""
        compass = self.compass["core_scores"]
        rows = []
        for c in self.case.cores:
            a, t = self.core(c), self.needle_track(c)
            rows.append(
                {
                    "core_id": c.core_id,
                    "pnf_core": mean_in(a["pnf"], t),
                    "dino_core": mean_in(a["dino"], t),
                    "compass_core": compass.get(c.core_id, np.nan),
                }
            )
        return pd.DataFrame(rows).set_index("core_id")

    @cached_property
    def quality(self) -> dict:
        """Outline quality on the first frame of the first core (the frame the paper's quality inputs were measured on):
        MicroSegNet's gland and NeedleNet's track."""
        frame = self.case.cores[0].frame(0)
        gland, needle = self.sp["microsegnet"].segment(frame), self.sp["needlenet"].segment(frame)
        g, n = mask_metrics(gland), mask_metrics(needle)
        return {
            "gland_valid": gland_is_valid(g),
            **{f"gland_{k}": v for k, v in g.items()},
            **{f"needle_{k}": v for k, v in n.items() if k != "centroid_offset_frac"},
            "needle_overlaps_gland_frac": round(float((gland & needle).sum() / max(needle.sum(), 1)), 4),
        }

    # ── study ────────────────────────────────────────────────────────────────────────────────────────────────────────
    @cached_property
    def compass(self) -> dict:
        if self.case.sweep is None:
            return {"patient_risk": np.nan, "core_scores": {}}
        return self.sp["compass"].predict(self.case)

    # ── sweep ────────────────────────────────────────────────────────────────────────────────────────────────────────
    @cached_property
    def sweep_frames(self) -> list[dict]:
        """Every sweep frame: roll (on the cores' reference), gland coverage, both maps' max / p90 / mean inside the gland,
        hot spots of both maps, and quarter-resolution maps and gland for the later steps."""
        sw = self.case.sweep
        out = []
        for k, roll in enumerate(sw.roll_aligned):
            frame = sw.frame(k)
            h, w = frame.shape
            small = (h // 4, w // 4)
            gland = self.sp["microsegnet"].segment(frame)
            pnf, dino = self.sp["prostnfound"].heatmap(frame, self.case.age, self.case.psa), self.sp["dinov3"].heatmap(frame)
            stats = lambda m: (
                {"max": float(m[gland].max()), "p90": float(np.percentile(m[gland], 90)), "mean": float(m[gland].mean())}
                if gland.any()
                else None
            )
            out.append(
                {
                    "k": k,
                    "roll": float(roll),
                    "shape": (h, w),
                    "gland_cov": float(gland.mean()),
                    "pnf": stats(pnf),
                    "dino": stats(dino),
                    "pnf_spots": hot_spots(pnf, gland),
                    "dino_spots": hot_spots(dino, gland),
                    "pnf_small": resize_map(pnf, small),
                    "dino_small": resize_map(dino, small),
                    "gland_small": resize_map(gland.astype(np.float32), small) > 0.5,
                }
            )
        return out

    def nearest_sweep_frame(self, core, max_angle: float, depth_tol_mm: float = 1.0):
        """The sweep frame closest in roll to the core, at the same imaging depth, within `max_angle` degrees (or None)."""
        sw = self.case.sweep
        if sw is None or abs(sw.depth_mm - core.depth_mm) >= depth_tol_mm:
            return None
        d = np.abs(sw.roll_aligned - core.roll_deg)
        k = int(np.argmin(d))
        return k if d[k] <= max_angle else None

    def registration_shift_mm(self, core) -> float:
        """Shift along the probe between the core's frame and the nearest same-depth sweep frame within 5 degrees (gives
        apex / mid / base); NaN without such a frame."""
        k = self.nearest_sweep_frame(core, 5.0)
        if k is None:
            return np.nan
        frame = self.case.sweep.frame(k)
        _, _, dx = register_frames(
            self.core(core)["frame"], self.core(core)["mm_per_px"], frame, self.case.sweep.mm_per_px(frame.shape)
        )
        return dx * GRID_MM

    @cached_property
    def gland_volume_ml(self) -> float:
        """Prostate volume from the sweep's gland outlines (Pappus): a gland slice of area A whose centroid is r from the probe
        axis sweeps A * r * dtheta; frames pooled on 2-degree roll bins, r = 10 mm probe radius + centroid depth. A relative
        size signal for the pseudo PSA density, not a measurement."""
        sw = self.case.sweep
        if sw is None:
            return np.nan
        rows = []
        for f in self.sweep_frames:
            g = f["gland_small"]
            px = sw.depth_mm / g.shape[0]
            n = float(g.sum())
            cy = (g * ((np.arange(g.shape[0]) + 0.5) * px)[:, None]).sum() / n if n else 0.0
            rows.append({"bin": np.floor(f["roll"] / VOLUME_BIN_DEG), "ar": n * px**2 * (VOLUME_PROBE_RADIUS_MM + cy)})
        v = pd.DataFrame(rows).groupby("bin").ar.mean().sum() * np.deg2rad(VOLUME_BIN_DEG) / 1000
        return float(v) if v > 0 else np.nan

    @property
    def sweep_statistics(self) -> list[dict]:
        """Per sweep frame: roll, gland coverage and each map's p90 inside the gland (what the window scores read)."""
        return (
            [
                {
                    "roll": f["roll"],
                    "gland_cov": f["gland_cov"],
                    "pnf_p90": f["pnf"]["p90"] if f["pnf"] else None,
                    "dino_p90": f["dino"]["p90"] if f["dino"] else None,
                }
                for f in self.sweep_frames
            ]
            if self.case.sweep
            else []
        )

    def window_scores(self, angle: float, z: dict) -> dict:
        return window_scores(self.sweep_statistics, angle, z)


def window_scores(frames: list[dict], angle: float, z: dict, window: float = 10.0) -> dict:
    """The sweep at a core's probe angle: over the frames within +-`window` degrees that show the gland, the max of each map's
    p90 inside the gland and of their combination (mean of the two after standardising with the cohort's frame statistics
    `z` = {"pnf": (mean, sd), "dino": (mean, sd)})."""
    fr = [
        f for f in frames if f["gland_cov"] >= MIN_SWEEP_GLAND and f["pnf_p90"] is not None and abs(f["roll"] - angle) <= window
    ]
    if not fr:
        return {"s_pnf_w10": np.nan, "s_dino_w10": np.nan, "s_comb_w10": np.nan}
    p, d = np.array([f["pnf_p90"] for f in fr]), np.array([f["dino_p90"] for f in fr])
    comb = ((p - z["pnf"][0]) / z["pnf"][1] + (d - z["dino"][0]) / z["dino"][1]) / 2
    return {"s_pnf_w10": float(p.max()), "s_dino_w10": float(d.max()), "s_comb_w10": float(comb.max())}
