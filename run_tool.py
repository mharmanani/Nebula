"""Run one specialist on its own on one case.

    python3 run_tool.py --tool prostnfound --modality frame --case /path/to/case [--core ID] [--frame K]
    python3 run_tool.py --tool microsegnet --modality sweep --case /path/to/case
    python3 run_tool.py --tool compass --case /path/to/case

Modalities: frame (one frame of a core; its fire frame by default), cine (every frame of a core), sweep (every sweep frame),
clinical (age and PSA). Compass reads the whole study: study or all (the default: the sweep, every core and age / PSA, as
trained), sweep (the sweep alone) or frame (the cores alone, or the one given by --core). Prints a summary per frame as JSON
and, with --out, saves the masks / maps as .npy.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import load_config
from data.OPTIMUM import Case
from tools.measure import mask_metrics
from tools.registry import SPECIALISTS, build_specialists

MODALITIES = ("frame", "cine", "sweep", "study", "all", "clinical")
NOT_SUPPORTED = {("needlenet", "sweep"): "no needle is fired during the sweep"}


def frames_of(case, modality, core_id, frame):
    if modality == "sweep":
        return [(f"sweep/{k}", case.sweep.frame(k)) for k in range(len(case.sweep))]
    core = case.core(core_id) if core_id else case.cores[0]
    idx = [core.fire_frame if frame is None else frame] if modality == "frame" else range(len(core.frame_paths))
    return [(f"{core.core_id}/{i}", core.frame(i)) for i in idx]


def run_on_frame(name, tool, frame, case):
    if name in ("microsegnet", "needlenet"):
        out = tool.segment(frame)
        return out, mask_metrics(out)
    out = tool.heatmap(frame, case.age, case.psa) if name == "prostnfound" else tool.heatmap(frame)
    return out, {"max": round(float(out.max()), 4), "mean": round(float(out.mean()), 4)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tool", choices=list(SPECIALISTS) + ["clinical"], required=True)
    ap.add_argument("--modality", choices=MODALITIES, help="required, except for compass (default: study) and clinical")
    ap.add_argument("--case", required=True)
    ap.add_argument("--core")
    ap.add_argument("--frame", type=int)
    ap.add_argument("--out", help="folder for the .npy masks / maps")
    a = ap.parse_args()
    case = Case.load(a.case)
    if a.tool == "clinical" or a.modality == "clinical":
        print(json.dumps({"age": case.age, "psa": case.psa}))
        return
    if a.modality is None:
        if a.tool != "compass":
            ap.error(f"--modality is required for {a.tool}")
        a.modality = "study"
    if a.modality not in SPECIALISTS[a.tool].modalities:
        raise SystemExit(
            f"{a.tool} does not take a {a.modality}: {NOT_SUPPORTED.get((a.tool, a.modality), 'supported: ' + ', '.join(SPECIALISTS[a.tool].modalities))}"
        )
    tool = build_specialists(load_config(), [a.tool])[a.tool]
    if a.tool == "compass":
        if a.modality == "sweep":
            case.cores = []
        elif a.modality == "frame":
            case.sweep = None
            case.cores = [case.core(a.core)] if a.core else case.cores
        print(json.dumps(tool.predict(case), indent=1))
        return
    out = Path(a.out) if a.out else None
    for key, frame in frames_of(case, a.modality, a.core, a.frame):
        result, summary = run_on_frame(a.tool, tool, frame, case)
        print(json.dumps({"frame": key, **summary}))
        if out:
            (out / key).parent.mkdir(parents=True, exist_ok=True)
            np.save(out / f"{key}.npy", result)


if __name__ == "__main__":
    main()
