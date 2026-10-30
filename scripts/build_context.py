"""Build the labelled context of the composers and the sweep scorer from your own cohort and specialists.

    python3 scripts/build_context.py --out /path/to/context [--splits splits/optimum_5fold_patient_splits.json]
                                     [--exclude splits/excluded_patients.json]

1. The scanner's usual needle path per centre: the mean annotated needle track of the cores of the other folds (and of all
   cores, for new patients).
2. Every case through the specialists (cached per case in <out>/cases/): the patient composer's inputs, per core the core
   composer's inputs and the live-frame inputs of the targeting composer (with the usual path of the other folds), the sweep
   frame statistics, and the sweep hot spots with their labels from the cores.
3. Tables: the sweep window scores (standardised with the cohort's sweep frames) and the anatomy inputs of every core, each
   with the calibration of the other folds, so that every row is out of fold as in the paper.
4. Thresholds per fold, fixed by inner cross-validation on that fold's context: patient = Youden; core = the top 30% of cores
   (the operators' biopsy rate in the paper's cohort). The targeting threshold is given at inference (run.py --threshold,
   default 0.14).
Labels are read here only (labels.json); never at inference.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from composer.context import fold_key
from composer.features import (
    CORE_FEATURES,
    PATIENT_FEATURES,
    anatomy_features,
    frame_features,
    patient_features,
    specialist_outputs,
)
from composer.tabpfn import FLAGGED_SHARE, TabPFNComposer, rate_threshold, youden_threshold
from config import load_config
from data.OPTIMUM import Cohort
from tools.analysis import MIN_SWEEP_GLAND, CaseAnalysis, window_scores
from tools.location import CoreLocator
from tools.measure import resize_mask
from tools.registry import build_specialists
from tools.sweep import candidates, weak_labels

DOWNSAMPLE = 4


def guide_paths(cohort: Cohort, keys) -> dict:
    """{(key, centre): mean annotated track} on a quarter-resolution grid, from the cores outside fold `key`."""
    tracks = {}
    for case in cohort:
        for c in case.cores:
            m = c.needle_mask()
            if m is not None:
                tracks.setdefault(case.center, []).append((cohort.fold_of(case.case_id), m))
    out = {}
    for center, tr in tracks.items():
        h, w = tr[0][1].shape
        shape = (int(np.ceil(h / DOWNSAMPLE)), int(np.ceil(w / DOWNSAMPLE)))
        for key in keys:
            sel = [resize_mask(m, shape) for f, m in tr if key == "all" or f != int(key)]
            if sel:
                out[f"{key}_{center}"] = np.mean(sel, 0).astype(np.float32)
    return out


def case_record(case, fold, specialists, paths) -> dict:
    """Everything the context needs from one case (no thresholds yet)."""
    a = CaseAnalysis(case, specialists)
    guide = paths.get(f"{fold_key(fold)}_{case.center}")
    outputs = specialist_outputs(a)
    scores = a.core_scores
    patient = {"case_id": case.case_id, "center": case.center, "fold": fold, "label": case.cspca, **patient_features(outputs)}
    cores = [
        {
            "core_id": c.core_id,
            "case_id": case.case_id,
            "center": case.center,
            "fold": fold,
            "label": c.cspca,
            "primus": (c.labels or {}).get("primus"),
            **scores.loc[c.core_id, CORE_FEATURES].to_dict(),
            **frame_features(a, c, guide),
            "angle": c.roll_deg,
            "depth_mm": c.depth_mm,
            "location": c.location,
            "reg_shift_mm": a.registration_shift_mm(c),
        }
        for c in case.cores
    ]
    spots = weak_labels(a, candidates(a)).assign(case_id=case.case_id, fold=fold) if case.sweep is not None else pd.DataFrame()
    return {"patient": patient, "cores": cores, "sweep": a.sweep_statistics, "hotspots": spots.to_dict("records")}


def anatomy_and_sweep(cores: pd.DataFrame, sweeps: dict, z: dict) -> pd.DataFrame:
    """Sweep window scores and anatomy inputs of every core, with the calibration of the other folds."""
    rows = []
    for f, held in cores.groupby("fold"):
        other = cores[cores.fold != f]
        locator = CoreLocator(other)
        for r in held.itertuples():
            same = other[other.center == r.center]
            med = float((same if len(same) else other).angle.abs().median())
            rows.append(
                {
                    "core_id": r.core_id,
                    **window_scores(sweeps.get(r.case_id, []), r.angle, z),
                    **anatomy_features(r.angle, r.depth_mm, r.reg_shift_mm, locator, med),
                }
            )
    return cores.merge(pd.DataFrame(rows), on="core_id")


def thresholds(patients, cores):
    """Per fold key: the patient (Youden) and core (top 30% of cores) thresholds on inner-CV out-of-fold scores."""
    out = {}
    for key in [str(f) for f in sorted(patients.fold.unique())] + ["all"]:
        P = patients if key == "all" else patients[patients.fold != int(key)]
        C = (cores if key == "all" else cores[cores.fold != int(key)]).dropna(subset=["label"])
        p = TabPFNComposer(PATIENT_FEATURES, P[PATIENT_FEATURES].values.astype(float), P.label.values.astype(int)).inner_oof()
        c = TabPFNComposer(CORE_FEATURES, C[CORE_FEATURES].values.astype(float), C.label.values.astype(int)).inner_oof()
        out[key] = {"patient": youden_threshold(P.label.values, p), "core": rate_threshold(c, FLAGGED_SHARE)}
        print(key, out[key], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=str(ROOT / "splits" / "optimum_5fold_patient_splits.json"))
    ap.add_argument("--exclude", default=str(ROOT / "splits" / "excluded_patients.json"))
    a = ap.parse_args()
    cfg = load_config()
    out = Path(a.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    cohort = Cohort(cfg["cohort"], a.splits)
    excluded = set(json.loads(Path(a.exclude).read_text())["case_ids"]) if a.exclude and Path(a.exclude).exists() else set()
    ids = [c for c in cohort.case_ids if c not in excluded and cohort.fold_of(c) is not None]
    keys = [str(f) for f in sorted({cohort.fold_of(c) for c in ids})] + ["all"]

    paths = guide_paths(cohort, keys)
    np.savez_compressed(out / "guide_paths.npz", **paths)
    specialists = build_specialists(cfg)
    records = {}
    for i, cid in enumerate(ids):
        f = out / "cases" / f"{cid}.json"
        if not f.exists():
            f.write_text(json.dumps(case_record(cohort.case(cid), cohort.fold_of(cid), specialists, paths), default=float))
        records[cid] = json.loads(f.read_text())
        print(f"[{i + 1}/{len(ids)}] {cid}", flush=True)

    patients = pd.DataFrame([r["patient"] for r in records.values()]).dropna(subset=["label"])
    frames = pd.DataFrame([s for r in records.values() for s in r["sweep"]])
    frames = frames[(frames.gland_cov >= MIN_SWEEP_GLAND) & frames.pnf_p90.notna()]
    z = {
        "pnf": [float(frames.pnf_p90.mean()), float(frames.pnf_p90.std())],
        "dino": [float(frames.dino_p90.mean()), float(frames.dino_p90.std())],
    }
    cores = anatomy_and_sweep(
        pd.DataFrame([c for r in records.values() for c in r["cores"]]), {k: r["sweep"] for k, r in records.items()}, z
    )
    hotspots = pd.DataFrame([s for r in records.values() for s in r["hotspots"]])
    patients.to_csv(out / "patients.csv", index=False)
    cores.to_csv(out / "cores.csv", index=False)
    hotspots.to_csv(out / "hotspots.csv", index=False)
    (out / "calibration.json").write_text(
        json.dumps({"sweep_z": z, "thresholds": thresholds(patients, cores)}, indent=1)
    )
    print(f"context: {len(patients)} patients, {len(cores)} cores, {len(hotspots)} sweep hot spots -> {out}")


if __name__ == "__main__":
    main()
