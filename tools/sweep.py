"""Candidate targets on the rotational sweep: every hot spot of either map inside the gland, described by sweep-only
features, scored by experience (TabPFN-3 in context over previous patients' labelled hot spots), and placed on the cores.

A previous patient's hot spot is labelled through the cores: the cores at the same imaging depth within 10 degrees are
registered with the hot spot's sweep frame; positive if a csPCa core's track passes within 5 mm, negative if only benign
tracks do, unlabelled if no core sampled it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from .analysis import MIN_SWEEP_GLAND
from .registration import GRID_MM, register_frames, to_grid

FEATURES = [
    "is_dino",
    "peak",
    "hot_mean",
    "r_mm",
    "blob_frac",
    "in_gland",
    "other_map",
    "depth_mm",
    "along_gland",
    "abs_angle",
    "side",
    "gland_cov",
    "persistence",
    "rank_in_frame",
]
N_TARGETS, MIN_SEPARATION_DEG, PLACE_WINDOW_DEG, LABEL_DISTANCE_MM = 5, 10.0, 10.0, 5.0
MAX_CONTEXT = 5000


def candidates(analysis) -> pd.DataFrame:
    """Every hot spot of both maps on the sweep frames that show the gland, with its features."""
    sw = analysis.case.sweep
    if sw is None:
        return pd.DataFrame(columns=FEATURES)
    frames = analysis.sweep_frames
    rows = []
    for f in frames:
        if f["gland_cov"] < MIN_SWEEP_GLAND:
            continue
        mm = sw.depth_mm / f["shape"][0]  # isotropic, as in the paper
        gx = np.nonzero(f["gland_small"].any(0))[0]
        x0, x1 = (gx.min() * 4, gx.max() * 4) if len(gx) else (0, f["shape"][1])
        for src, other in (("pnf", "dino_small"), ("dino", "pnf_small")):
            for i, s in enumerate(f[f"{src}_spots"]):
                o = f[other]
                oy, ox = min(int(s["cy"] // 4), o.shape[0] - 1), min(int(s["cx"] // 4), o.shape[1] - 1)
                persistence = sum(
                    any(np.hypot((t["cx"] - s["cx"]) * mm, (t["cy"] - s["cy"]) * mm) <= 3 for t in g[f"{src}_spots"])
                    for g in frames
                    if g["k"] != f["k"] and abs(g["roll"] - f["roll"]) <= 3
                )
                rows.append(
                    {
                        "k": f["k"],
                        "angle": f["roll"],
                        "src": src,
                        "is_dino": int(src == "dino"),
                        "rank_in_frame": i,
                        "peak": s["max_prob"],
                        "hot_mean": s["hot_mean"],
                        "r_mm": s["r_px"] * mm,
                        "blob_frac": s["blob_area_frac"],
                        "in_gland": s["frac_inside_gland"],
                        "other_map": float(o[oy, ox]),
                        "depth_mm": s["cy"] * mm,
                        "along_gland": (s["cx"] - x0) / max(x1 - x0, 1),
                        "abs_angle": abs(f["roll"]),
                        "side": int(f["roll"] > 0),
                        "gland_cov": f["gland_cov"],
                        "persistence": int(persistence),
                        "cx": s["cx"],
                        "cy": s["cy"],
                    }
                )
    return pd.DataFrame(rows, columns=["k", "angle", "src", *FEATURES, "cx", "cy"]).dropna(subset=FEATURES).reset_index(drop=True)


def top_distinct(df: pd.DataFrame, score: str, n: int = N_TARGETS) -> pd.DataFrame:
    """The best-scored candidates at least 10 degrees apart."""
    picks = []
    for i in np.argsort(-df[score].values):
        if all(abs(df.angle.values[i] - df.angle.values[j]) >= MIN_SEPARATION_DEG for j in picks):
            picks.append(i)
        if len(picks) == n:
            break
    return df.iloc[picks].reset_index(drop=True)


class ExperienceScorer:
    """TabPFN-3 in context over previous patients' labelled hot spots (stratified subsample to 5,000 if larger)."""

    def __init__(self, context_spots: pd.DataFrame, seed: int = 0):
        from composer.tabpfn import fit_tabpfn

        spots = context_spots[context_spots.label >= 0]
        X, y = spots[FEATURES].values.astype(float), spots.label.values.astype(int)
        if len(X) > MAX_CONTEXT:
            rng = np.random.default_rng(seed)
            keep = np.concatenate(
                [rng.choice(np.where(y == c)[0], min(MAX_CONTEXT // 2, (y == c).sum()), replace=False) for c in (0, 1)]
            )
            X, y = X[keep], y[keep]
        self.model, self.spots = fit_tabpfn(X, y, seed), spots
        mu, sd = spots[FEATURES].values.mean(0), spots[FEATURES].values.std(0) + 1e-9
        self._z = lambda a: (np.asarray(a, float) - mu) / sd

    def score(self, cands: pd.DataFrame) -> np.ndarray:
        from composer.tabpfn import positive_proba

        return positive_proba(self.model, cands[FEATURES].values.astype(float)) if len(cands) else np.zeros(0)

    def similar(self, cand: pd.Series, n: int = 5) -> pd.DataFrame:
        """The n most similar labelled hot spots of previous patients (Euclidean distance on standardised features)."""
        d = np.linalg.norm(self._z(self.spots[FEATURES].values) - self._z(cand[FEATURES].values), axis=1)
        return self.spots.iloc[np.argsort(d)[:n]]


def place_on_core(analysis, cand: pd.Series):
    """The core that sampled a sweep hot spot: same imaging depth, nearest roll within 10 degrees. Returns (core, distance in
    mm from the hot spot to that core's track after registration) or (None, None)."""
    case = analysis.case
    near = [
        c for c in case.cores if abs(c.roll_deg - cand.angle) <= PLACE_WINDOW_DEG and abs(c.depth_mm - case.sweep.depth_mm) < 1
    ]
    if not near:
        return None, None
    core = min(near, key=lambda c: abs(c.roll_deg - cand.angle))
    return core, track_distance_mm(analysis, core, cand)


def track_distance_mm(analysis, core, cand: pd.Series) -> float:
    """Distance from the hot spot's centre to the core's needle track, after registering the two frames."""
    sw = analysis.case.sweep
    frame = sw.frame(int(cand.k))
    smm = sw.mm_per_px(frame.shape)
    a = analysis.core(core)
    _, dy, dx = register_frames(a["frame"], a["mm_per_px"], frame, smm)
    track = to_grid(analysis.needle_track(core).astype(np.float32), a["mm_per_px"], order=0) > 0.5
    if not track.any():
        return np.nan
    dist = distance_transform_edt(~track) * GRID_MM
    yb, xb = int(round(cand.cy * smm[0] / GRID_MM)) + dy, int(round(cand.cx * smm[1] / GRID_MM)) + dx
    return float(dist[yb, xb]) if 0 <= yb < dist.shape[0] and 0 <= xb < dist.shape[1] else np.nan


def weak_labels(analysis, cands: pd.DataFrame) -> pd.DataFrame:
    """Labels of a labelled patient's hot spots for the context (build_context.py only): 1 / 0 / -1 as described above, the
    PRI-MUS of the nearest sampling core, and the distances."""
    case = analysis.case
    out = cands.copy()
    labels, primus, dist = [], [], []
    for _, s in cands.iterrows():
        near = [
            c
            for c in case.cores
            if abs(c.roll_deg - s.angle) <= PLACE_WINDOW_DEG and abs(c.depth_mm - case.sweep.depth_mm) < 1 and c.cspca is not None
        ]
        hits = [(track_distance_mm(analysis, c, s), c) for c in near]
        hits = [(d, c) for d, c in hits if d == d]
        within = [(d, c) for d, c in hits if d <= LABEL_DISTANCE_MM]
        labels.append(1 if any(c.cspca for _, c in within) else (0 if within else -1))
        nearest = min(within or hits, key=lambda h: h[0], default=None)
        primus.append((nearest[1].labels or {}).get("primus") if nearest else None)
        dist.append(nearest[0] if nearest else np.nan)
    return out.assign(label=labels, primus=primus, core_dist_mm=dist)
