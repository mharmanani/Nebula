"""One episode: the case, the workflow and what the coordinator has collected; one method per coordinator tool.

The workflow fixes what has been acquired. Post-biopsy review: the sweep, every core with its track, age and PSA.
Intra-procedural targeting (`core` = the core being aimed): the sweep, age, PSA and the core's live frame only -- its track,
the other cores and any pathology are never used. Tools whose input does not exist yet answer with the reason.
"""

from __future__ import annotations

import inspect
import re
from functools import cached_property

import numpy as np
import pandas as pd

from composer.features import (
    CORE_FEATURES,
    TARGETING_FEATURES,
    clinical_outputs,
    frame_features,
    patient_features,
    specialist_outputs,
    targeting_features,
)
from composer.tabpfn import TARGETING_THRESHOLD
from tools.lesion import find_lesion
from tools.measure import needle_geometry
from tools.sweep import candidates, place_on_core, top_distinct

from .wording import RECOMMENDATION, margin, risk_statement, threshold_note

INTRA_TOOLS = {
    "scan_sweep_targets",
    "similar_past_cases",
    "locate_core",
    "assess_gland",
    "frame_cancer_map",
    "second_opinion_map",
    "clinical_context",
    "target_lesion",
    "record_findings",
    "compose",
}
CORE_COMPOSER_TOOLS = ("frame_cancer_map", "second_opinion_map", "study_level_risk")
FRAME_TOOLS = ("assess_gland", "frame_cancer_map", "second_opinion_map")
LESION_NOTE = (
    "the hottest point of the average of both cancer maps inside the prostate outline; a candidate only -- it is the "
    "suggested lesion only if the composition decides that this core should be biopsied"
)


def rounded(x, digits: int = 4):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if x != x else round(float(x), digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    return x


class Session:
    def __init__(self, analysis, context, workflow: str, core_id: str | None = None, threshold: float = TARGETING_THRESHOLD):
        """`threshold`: intra-procedural targeting, biopsy if P(csPCa) >= threshold."""
        assert workflow in ("post", "intra") and (workflow == "post" or core_id)
        self.a, self.ctx, self.workflow, self.threshold = analysis, context, workflow, threshold
        self.case = analysis.case
        self.aimed = self.case.core(core_id) if core_id else None
        self.collected, self.findings, self.final = {}, [], None
        self.targets = None
        self.placed = self.opened = None
        self.forcing = False

    def call(self, name: str, arguments: dict | None = None) -> dict:
        """Run one coordinator tool (unknown arguments are ignored)."""
        blocked = self._unavailable(name)
        if blocked:
            return blocked
        method = getattr(self, name)
        accepted = inspect.signature(method).parameters
        return method(**{k: v for k, v in (arguments or {}).items() if k in accepted})

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _unavailable(self, name):
        if self.workflow == "intra" and name not in INTRA_TOOLS:
            why = "the needle has not been fired yet" if name == "locate_needle" else "it needs the taken biopsy cores"
            return {"error": f"not available while a core is being aimed: {why}"}
        return None

    def _core_name(self, core) -> str:
        return f"core {self.case.cores.index(core) + 1} ({core.core_id})"

    def _find_core(self, ref):
        ref = str(ref or "").strip()
        m = re.fullmatch(r"(?i)core\s*(\d+).*", ref)
        if m and 1 <= int(m.group(1)) <= len(self.case.cores):
            return self.case.cores[int(m.group(1)) - 1]
        return next((c for c in self.case.cores if c.core_id == ref or c.core_id in ref), None) if ref else None

    def _find_target(self, ref):
        m = re.search(r"T\s*(\d+)", str(ref or "").upper())
        i = int(m.group(1)) - 1 if m else -1
        return (f"T{i + 1}", self.targets.iloc[i]) if self.targets is not None and 0 <= i < len(self.targets) else (None, None)

    def _where(self, core) -> str:
        return self.ctx.locator.describe(core.roll_deg, self.a.registration_shift_mm(core))

    def _guide_path(self):
        return self.ctx.calibration_for(self.case.center)["guide_path"]

    def _lesion(self, core, with_track: bool) -> dict:
        a = self.a.core(core)
        les = find_lesion(
            a["pnf"], a["dino"], a["gland"], a["mm_per_px"], self._guide_path(), self.a.needle_track(core) if with_track else None
        )
        found = les.pop("gland_found")
        keys = ("needle_track_crosses_lesion", "lesion_centre_to_track_mm")
        out = {"candidate_lesion": {k: v for k, v in les.items() if k not in keys}, "note": LESION_NOTE}
        if not found:
            out["note"] += "; no prostate outline was found on this frame, so the whole frame was searched"
        out.update({k: les[k] for k in keys if k in les})
        return out

    @cached_property
    def core_risks(self) -> pd.Series:
        X = self.a.core_scores[CORE_FEATURES].values.astype(float)
        return pd.Series(self.ctx.core_composer.proba(X), index=self.a.core_scores.index)

    @cached_property
    def outputs(self) -> dict:
        """What each post-biopsy specialist tool returns, computed once per episode."""
        return specialist_outputs(self.a)

    @cached_property
    def live_frame(self) -> dict:
        """The live frame's inputs while a core is aimed, computed once per episode."""
        return frame_features(self.a, self.aimed, self._guide_path())

    # ── the sweep and quality assurance ──────────────────────────────────────────────────────────────────────────────
    def scan_sweep_targets(self):
        if self.case.sweep is None:
            return {"error": "no sweep available for this study"}
        if self.ctx.experience is None:
            return {"error": "no labelled hot spots in the context to score the candidates"}
        if self.targets is None:
            cands = candidates(self.a)
            if cands.empty:
                return {"error": "no suspicious region inside the gland on this sweep"}
            self.targets = top_distinct(cands.assign(score=self.ctx.experience.score(cands)), "score")
        zone = self.ctx.locator.zone
        return {
            "targets": [
                {
                    "id": f"T{i + 1}",
                    "sweep_frame": int(r.k) + 1,
                    "probe_angle_deg": round(float(r.angle), 1),
                    "side": "left" if r.angle < 0 else "right",
                    "zone": zone(r.angle),
                    "depth_mm": round(float(r.depth_mm), 1),
                    "radius_mm": round(float(r.r_mm), 1),
                    "second_opinion_at_spot": round(float(r.peak if r.is_dino else r.other_map), 3),
                    "frame_map_at_spot": round(float(r.other_map if r.is_dino else r.peak), 3),
                    "persistence_frames": int(r.persistence),
                    "experience_score": round(float(r.score), 3),
                }
                for i, r in self.targets.iterrows()
            ]
        }

    def similar_past_cases(self, target=None):
        tid, r = self._find_target(target)
        if r is None:
            return {"error": f"unknown target '{target}'; call scan_sweep_targets first and use one of its ids"}
        nn = self.ctx.experience.similar(r)
        past = [
            {
                "outcome": "cancer (GG>=2) core" if s.label == 1 else "benign core",
                "operator_primus": None if pd.isna(s.primus) else int(s.primus),
                "second_opinion_peak": round(float(s.peak if s.is_dino else s.other_map), 3),
                "radius_mm": round(float(s.r_mm), 1),
            }
            for s in nn.itertuples()
        ]
        return {
            "target": tid,
            "similar_past_hot_spots": past,
            "summary": f"{int(nn.label.sum())} of {len(nn)} similar past hot spots sat on a cancer core",
        }

    def locate_core(self):
        if self.workflow == "post":
            return {
                "error": "only while a core is being aimed (intra-procedural targeting); after the biopsy use place_on_cores and open_core"
            }
        if self.targets is None:
            return {"error": "no sweep targets yet: call scan_sweep_targets first"}
        c, depth_ok = self.aimed, abs(self.aimed.depth_mm - self.case.sweep.depth_mm) < 1
        on = [
            {"target": f"T{i + 1}", "angle_difference_deg": round(float(c.roll_deg - r.angle), 1)}
            for i, r in self.targets.iterrows()
            if depth_ok and abs(c.roll_deg - r.angle) <= 10
        ]
        return {
            "core_location": self._where(c),
            "probe_angle_deg": round(float(c.roll_deg), 1),
            "sweep_targets_on_this_core": on or "none: no sweep target lies at this core's probe angle and depth",
        }

    def place_on_cores(self, target=None):
        tid, r = self._find_target(target)
        if r is None:
            return {"error": f"unknown target '{target}'; call scan_sweep_targets first and use one of its ids"}
        core, dist = place_on_core(self.a, r)
        if core is None:
            return {
                "target": tid,
                "placed": False,
                "reason": "no biopsy core was taken at the same imaging depth within 10 degrees of this region: it was not sampled",
            }
        self.placed = core
        return {
            "target": tid,
            "placed": True,
            "matched_core": self._core_name(core),
            "core_location": self._where(core),
            "side_from_target_angle": "left" if r.angle < 0 else "right",
            "angle_difference_deg": round(float(core.roll_deg - r.angle), 1),
            "target_to_core_track_mm": None if dist != dist else round(float(dist), 1),
        }

    def open_core(self, core=None):
        c = self._find_core(core) if core else self.placed
        if c is None:
            return {
                "error": f"unknown core '{core}'; use the core named by place_on_cores or 'core N' (1-{len(self.case.cores)})"
            }
        self.opened = c
        s = self.a.core_scores
        a = self.a.core(c)
        rank = lambda col: f"{int((s[col] > s.loc[c.core_id, col]).sum()) + 1} of {int(s[col].notna().sum())}"
        return {
            "core": self._core_name(c),
            "core_location": self._where(c),
            "probe_angle_deg": round(float(c.roll_deg), 1),
            "frame_map_score_in_track": rounded(s.loc[c.core_id, "pnf_core"]),
            "frame_map_rank": rank("pnf_core"),
            "second_opinion_score_in_track": rounded(s.loc[c.core_id, "dino_core"]),
            "second_opinion_rank": rank("dino_core"),
            "gland": {k: a["gland_metrics"][k] for k in ("coverage_frac", "n_components")},
            "needle_track": needle_geometry(self.a.needle_track(c), a["mm_per_px"]),
        }

    def target_lesion(self, core=None):
        if self.workflow == "intra":
            return self._lesion(self.aimed, with_track=False)
        c = self._find_core(core) if core else self.opened
        if c is None:
            return {
                "error": f"unknown core '{core}'; use the core named by place_on_cores or 'core N' (1-{len(self.case.cores)})"
            }
        return {
            "core": self._core_name(c),
            **self._lesion(c, with_track=True),
            "shown_as_suggested_lesion": bool(self.core_risks[c.core_id] >= self.ctx.thresholds["core"]),
        }

    def record_findings(self, location="", core="", justification=""):
        self.findings.append(
            {"location": str(location)[:600], "core": str(core)[:600], "justification": str(justification)[:600]}
        )
        return {"recorded": True, "findings_so_far": len(self.findings)}

    # ── specialists ──────────────────────────────────────────────────────────────────────────────────────────────────
    def _collect(self, name):
        self.collected[name] = self.outputs[name]
        return rounded(self.outputs[name])

    def _specialist(self, name):
        return self._live_frame(name) if self.workflow == "intra" else self._collect(name)

    def _live_frame(self, name):
        f = self.live_frame
        r3 = lambda v: rounded(v, 3)
        if name == "assess_gland":
            out = {"prostate_in_view": bool(f["gland_valid"]), "prostate_area_fraction_of_frame": r3(f["gland_cov"])}
        else:
            p = "pnf" if name == "frame_cancer_map" else "dino"
            out = {
                "model": "ProstNFound+" if p == "pnf" else "DINOv3",
                "max_in_prostate": r3(f[f"{p}_max_gland"]),
                "mean_in_prostate": r3(f.get(f"{p}_mean_gland")),
                "suspicious_regions": int(f[f"{p}_n_cand"]),
                "strongest_region_score": r3(f[f"{p}_cand_hot"]),
                "along_guide_line_mean": r3(f[f"{p}_path_wmean"]),
                "along_guide_line_max": r3(f[f"{p}_path_max"]),
                "along_guide_line_p90": r3(f[f"{p}_path_p90"]),
            }
        self.collected[name] = out
        return out

    def assess_gland(self):
        return self._specialist("assess_gland")

    def locate_needle(self):
        return self._specialist("locate_needle")

    def frame_cancer_map(self):
        return self._specialist("frame_cancer_map")

    def second_opinion_map(self):
        return self._specialist("second_opinion_map")

    def study_level_risk(self):
        if self.case.sweep is None:
            return {"error": "no sweep available for this study"}
        return self._specialist("study_level_risk")

    def clinical_context(self):
        self.collected["clinical_context"] = clinical_outputs(self.a)  # age and PSA only: no other core is touched
        return rounded(self.collected["clinical_context"])

    # ── composer ─────────────────────────────────────────────────────────────────────────────────────────────────────
    def compose(self):
        return self._compose_intra() if self.workflow == "intra" else self._compose_post()

    def _compose_post(self):
        res = self.ctx.patient_composer.predict(patient_features(self.collected), "csPCa likely", "csPCa unlikely")
        res["risk_statement"] = risk_statement(res["risk"], res["threshold"], "in this patient", "decision threshold")
        missing = [t for t in CORE_COMPOSER_TOOLS if t not in self.collected]
        if missing:
            res["cores"] = {"error": f"core-level composition needs the outputs of {missing}"}
        else:
            t = self.ctx.thresholds["core"]
            res["cores"] = [
                {
                    "core": self._core_name(self.case.core(cid)),
                    "location": self._where(self.case.core(cid)),
                    "composite_risk": round(float(p), 4),
                    "flagged": bool(p >= t),
                }
                for cid, p in self.core_risks.sort_values(ascending=False).items()
            ]
        self.final = {"kind": "compose_post", **res}
        return res

    def _compose_intra(self):
        need = [t for t in FRAME_TOOLS if t not in self.collected]
        if need and not self.forcing:
            return {"error": f"the composition for this frame needs {need} first"}
        f = targeting_features(self.a, self.aimed, self.ctx)
        p = float(self.ctx.targeting_composer.proba(np.array([[f.get(k, np.nan) for k in TARGETING_FEATURES]], float))[0])
        t = float(self.threshold)
        yes = p >= t
        res = {
            "probability": round(p, 4),
            "threshold": round(t, 4),
            "decision": "biopsy" if yes else "no biopsy",
            "margin": margin(p, t),
            "risk_statement": risk_statement(p, t),
            "recommendation": RECOMMENDATION[yes],
            "suggested_lesion": self._lesion(self.aimed, with_track=False)["candidate_lesion"] if yes else None,
        }
        note = threshold_note(p, t)
        if note:
            res["threshold_note"] = note
        self.final = {"kind": "compose_intra", **res, "inputs_missing": need}
        return res
