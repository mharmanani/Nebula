"""Where the tools meet: the specialists are built from your weights (one model per tool), and the 14 tools the coordinator
can call are named, described and bound to an episode.

Specialists  microsegnet (gland), needlenet (needle track), prostnfound (cancer map), dinov3 (second-opinion map),
             compass (study level)
Coordinator  scan_sweep_targets, similar_past_cases, locate_core, place_on_cores, open_core, target_lesion, record_findings,
             assess_gland, locate_needle, frame_cancer_map, second_opinion_map, study_level_risk, clinical_context, compose
The descriptions are those the paper's coordinator received.
"""

from __future__ import annotations

from .base import Tool
from .compass import Compass
from .dinov3 import DINOv3
from .microsegnet import MicroSegNet
from .needlenet import NeedleNet
from .prostnfound import ProstNFound

SPECIALISTS = {
    "microsegnet": MicroSegNet,
    "needlenet": NeedleNet,
    "prostnfound": ProstNFound,
    "dinov3": DINOv3,
    "compass": Compass,
}


def build_specialists(cfg: dict, names=None) -> dict:
    """The specialists named (all by default) from the config: cfg["weights"] = {specialist: checkpoint} (microsegnet may be
    null: public weights) and cfg["device"]."""
    return {
        n: SPECIALISTS[n](cfg["weights"].get(n), device=cfg.get("device", "cpu"))
        for n in (names or SPECIALISTS)
    }


NEEDS_CORES = " Needs the biopsy cores."
DESCRIPTIONS = {
    "scan_sweep_targets": "Scans the rotational sweep with both cancer-likelihood maps and returns the most promising distinct candidate "
    "targets (ids T1-T5), ranked by an experience score (TabPFN-3 in context over previous patients' labelled hot "
    "spots). Each has its sweep frame, probe angle, side and zone (from the angle), an estimated apex/mid/base "
    "level, depth and size, both maps' values at the spot and how many neighbouring sweep frames show it "
    "(persistence). Needs the sweep.",
    "similar_past_cases": "For one candidate (argument `target`, e.g. T2): the five most similar hot spots seen in previous patients, "
    "whether a biopsy core there turned out to contain clinically significant cancer or was benign, and the "
    "operator's PRI-MUS score for that core.",
    "locate_core": "While a core is being aimed: the anatomical location of this core estimated by Nebula (side and medial/lateral "
    "from the probe angle, apex/mid/base from image registration with the sweep) and which sweep targets lie at its "
    "probe angle and imaging depth. Needs scan_sweep_targets first.",
    "place_on_cores": "Places one sweep target (argument `target`, e.g. T2) on the biopsy cores: the core taken at the same imaging depth "
    "within 10 degrees of the target's probe angle, its recorded anatomical location, the angle difference and the "
    "distance from the target to that core's needle track after label-free image registration -- or that no core "
    "sampled the region." + NEEDS_CORES,
    "open_core": "Opens one biopsy core (argument `core`, e.g. the core returned by place_on_cores): its frame-level cancer-map score "
    "and second-opinion score along the needle track, how each ranks among the patient's cores, the gland outline on "
    "that frame and the needle-track geometry." + NEEDS_CORES,
    "target_lesion": "The lesion on a core's frame: the hottest point of the average of both cancer maps inside the prostate outline, as "
    "a 4 mm disc (depth, both maps' values, distance from the guide line). After the biopsy (argument `core`): "
    "also whether that core's needle track crosses it. While a core is aimed: the live frame, no argument. A "
    "candidate only: it is the suggested lesion only if the composition decides the core should be biopsied.",
    "record_findings": "Records one finding for the quality-assurance report: `location` (anatomical location of the region you followed), "
    "`core` (the core that sampled it, or 'none'), `justification` (why you followed it and what the opened core and "
    "the lesion target showed). Does not end the task.",
    "assess_gland": "MicroSegNet prostate outline: whether the prostate is adequately in view and outlined. After the biopsy: on the "
    "cores' frames (coverage, components, centroid offset, edge contact). While a core is aimed: on the live frame. "
    "Anatomy and image quality only -- says nothing about cancer.",
    "locate_needle": "Segments the biopsy needle track on the reviewed frame (NeedleNet) and reports where the core sampled: track "
    "coverage, overlap with the gland outline, edge contact. Anatomy only -- says nothing about cancer." + NEEDS_CORES,
    "frame_cancer_map": "ProstNFound+ cancer-likelihood map (0-1). After the biopsy: along each core's needle track (patient-level "
    "score and per-core summaries). While a core is aimed: on the live frame -- maximum and mean inside the "
    "prostate, suspicious regions, and the map along the scanner's needle guide line.",
    "second_opinion_map": "DINOv3 cancer-likelihood map, a different model family trained on another cohort; the same summaries as "
    "frame_cancer_map (after the biopsy and while a core is aimed). Useful as a cross-check.",
    "study_level_risk": "Compass: study-level clinically-significant-cancer probability fusing the full rotational sweep, all biopsy cores "
    "and clinical prompts. The strongest single number, but poorly calibrated at the extremes." + NEEDS_CORES,
    "clinical_context": "Returns the patient's age and PSA.",
    "compose": "The composition (TabPFN-3, in context over previous patients). Do not produce a number yourself. After the biopsy: "
    "study-level csPCa risk and decision over every specialist output collected (missing inputs are imputed), and each "
    "core's composite risk and whether it is flagged. While a core is aimed: the probability of significant cancer "
    "along the guide line from the frame's gland and both maps, and the decision -- biopsy this core or not -- with the "
    "suggested lesion when the decision is biopsy. Ends the task.",
}
ARGUMENTS = {
    "similar_past_cases": {"target": {"type": "string"}},
    "place_on_cores": {"target": {"type": "string"}},
    "open_core": {"core": {"type": "string"}},
    "target_lesion": {"core": {"type": "string"}},
    "record_findings": {"location": {"type": "string"}, "core": {"type": "string"}, "justification": {"type": "string"}},
}
INPUT_KIND = {
    "scan_sweep_targets": "sweep",
    "similar_past_cases": "outputs",
    "locate_core": "frame",
    "place_on_cores": "study",
    "open_core": "frame",
    "target_lesion": "frame",
    "record_findings": "outputs",
    "assess_gland": "frame",
    "locate_needle": "frame",
    "frame_cancer_map": "frame",
    "second_opinion_map": "frame",
    "study_level_risk": "study",
    "clinical_context": "clinical",
    "compose": "outputs",
}
COORDINATOR_TOOLS = list(DESCRIPTIONS)
REPEATABLE = ("similar_past_cases", "place_on_cores", "open_core", "target_lesion", "record_findings")


def coordinator_tools(session) -> dict[str, Tool]:
    """The coordinator's tools, bound to one episode (`agent.session.Session`)."""
    return {
        n: Tool(n, DESCRIPTIONS[n], INPUT_KIND[n], run=lambda _n=n, **kw: session.call(_n, kw), arguments=ARGUMENTS.get(n))
        for n in COORDINATOR_TOOLS
    }
