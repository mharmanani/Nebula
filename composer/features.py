"""The composers' inputs, built from the specialists' outputs on one case.

patient (post-biopsy review, 14): Compass's study risk; ProstNFound+ (max over cores) and DINOv3 (90th percentile over
         cores) of the mean map inside each core's needle track; outline quality of the gland and the needle; age, PSA and
         a pseudo PSA density (PSA / prostate volume estimated from the sweep's gland outlines)
core     (post-biopsy review, 3): the core's ProstNFound+ and DINOv3 track means and Compass's per-core score
targeting (a core while it is aimed, 23): the live frame (both maps inside the gland and along the scanner's usual needle
         path, the gland), the sweep at the core's probe angle, and the anatomy (side, |roll|, zone, level, depth)
"""

from __future__ import annotations

import numpy as np

from tools.measure import along_path, gland_is_valid, hot_spots

PATIENT_FEATURES = [
    "compass_pat",
    "pnf_pat",
    "dino_p90",
    "gland_centroid_offset_frac",
    "gland_coverage_frac",
    "gland_n_components",
    "gland_touches_edge",
    "gland_valid",
    "needle_coverage_frac",
    "needle_touches_edge",
    "needle_overlaps_gland_frac",
    "age",
    "psa",
    "pseudo_psad",
]
CORE_FEATURES = ["pnf_core", "dino_core", "compass_core"]
FRAME_FEATURES = [
    "pnf_max_gland",
    "pnf_mean_gland",
    "dino_max_gland",
    "gland_valid",
    "gland_cov",
    "pnf_n_cand",
    "pnf_cand_hot",
    "dino_n_cand",
    "dino_cand_hot",
    "pnf_path_wmean",
    "pnf_path_max",
    "pnf_path_p90",
    "dino_path_wmean",
    "dino_path_max",
    "dino_path_p90",
]
SWEEP_FEATURES = ["s_pnf_w10", "s_dino_w10", "s_comb_w10"]
ANATOMY_FEATURES = ["side", "absroll_c", "zone_lat", "level", "depth"]
TARGETING_FEATURES = FRAME_FEATURES + SWEEP_FEATURES + ANATOMY_FEATURES
LEVEL_CODE = {"apex": 0.0, "mid": 1.0, "base": 2.0}


def core_statistics(prefix: str, values) -> dict:
    v = np.asarray([x for x in values if x == x], float)
    if not len(v):
        return {f"{prefix}_max": np.nan, f"{prefix}_mean": np.nan, f"{prefix}_p90": np.nan, f"{prefix}_n": 0.0}
    return {
        f"{prefix}_max": float(v.max()),
        f"{prefix}_mean": float(v.mean()),
        f"{prefix}_p90": float(np.percentile(v, 90)),
        f"{prefix}_n": float(len(v)),
    }


def clinical_outputs(analysis) -> dict:
    """Age, PSA and the pseudo PSA density (PSA / prostate volume estimated from the sweep's gland outlines)."""
    case, volume = analysis.case, analysis.gland_volume_ml
    return {
        "age": case.age,
        "psa": case.psa,
        "pseudo_psad": case.psa / volume if case.psa is not None and volume == volume else np.nan,
    }


def specialist_outputs(analysis) -> dict:
    """What each post-biopsy specialist tool returns (full precision); the patient composer reads its inputs from these."""
    s, q = analysis.core_scores, analysis.quality
    pnf, dino, comp = core_statistics("pnf_core", s.pnf_core), core_statistics("dino_core", s.dino_core), s.compass_core.dropna()
    return {
        "assess_gland": {
            k: q[k]
            for k in (
                "gland_valid",
                "gland_coverage_frac",
                "gland_n_components",
                "gland_centroid_offset_frac",
                "gland_touches_edge",
            )
        },
        "locate_needle": {
            k: q[k] for k in ("needle_coverage_frac", "needle_touches_edge", "needle_overlaps_gland_frac", "needle_n_components")
        },
        "frame_cancer_map": {
            "pnf_pat": pnf["pnf_core_max"],
            "pnf_core_mean": pnf["pnf_core_mean"],
            "pnf_core_p90": pnf["pnf_core_p90"],
            "pnf_core_n": pnf["pnf_core_n"],
        },
        "second_opinion_map": {
            "dino_p90": dino["dino_core_p90"],
            "dino_pat": dino["dino_core_max"],
            "dino_mean": dino["dino_core_mean"],
        },
        "study_level_risk": {
            "compass_pat": analysis.compass["patient_risk"],
            "compass_core_max": float(comp.max()) if len(comp) else np.nan,
            "compass_core_frac_pos": float((comp > 0).mean()) if len(comp) else np.nan,
        },
        "clinical_context": clinical_outputs(analysis),
    }


def patient_features(collected: dict) -> dict:
    """The patient composer's inputs among the outputs collected so far (missing ones are imputed by the composer)."""
    return {
        k: float(v) for out in collected.values() for k, v in out.items() if k in PATIENT_FEATURES and v is not None and v == v
    }


def frame_features(analysis, core, guide_path) -> dict:
    """The live frame (no needle): both maps inside the gland, their hot spots, the gland, and both maps along the usual
    needle path of the centre (`guide_path`, the mean annotated track of the context's cores)."""
    a = analysis.core(core)
    gland = a["gland"]
    out = {}
    for name in ("pnf", "dino"):
        m = a[name]
        spots = hot_spots(m, gland)
        out[f"{name}_max_gland"] = float(m[gland].max()) if gland.any() else np.nan
        out[f"{name}_mean_gland"] = float(m[gland].mean()) if gland.any() else np.nan
        out[f"{name}_n_cand"] = float(len(spots))
        out[f"{name}_cand_hot"] = max((s["hot_mean"] for s in spots), default=0.0)
        path = along_path(m, guide_path) if guide_path is not None else {"wmean": np.nan, "max": np.nan, "p90": np.nan}
        out.update({f"{name}_path_{k}": v for k, v in path.items()})
    out["gland_valid"], out["gland_cov"] = float(gland_is_valid(a["gland_metrics"])), a["gland_metrics"]["coverage_frac"]
    return out  # FRAME_FEATURES + dino_mean_gland (shown, not a composer input)


def anatomy_features(angle: float, depth_mm: float, reg_shift_mm: float, locator, absroll_median: float) -> dict:
    """Side (sign of the roll), |roll| centred on the centre's median, lateral zone, level (apex 0, mid 1, base 2; unknown
    -> imputed) and imaging depth."""
    return {
        "side": float(np.sign(angle)),
        "absroll_c": abs(angle) - absroll_median,
        "zone_lat": float(locator.zone(angle) == "lateral"),
        "level": LEVEL_CODE.get(locator.level(reg_shift_mm), np.nan),
        "depth": depth_mm,
    }


def targeting_features(analysis, core, context) -> dict:
    cal = context.calibration_for(analysis.case.center)
    f = frame_features(analysis, core, cal["guide_path"])
    f.update(analysis.window_scores(core.roll_deg, context.sweep_z))
    f.update(
        anatomy_features(
            core.roll_deg, core.depth_mm, analysis.registration_shift_mm(core), context.locator, cal["absroll_median"]
        )
    )
    return f
