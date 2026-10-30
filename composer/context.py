"""The labelled context the composers and the sweep scorer read: previous patients' tool outputs and outcomes, built from
your own cohort and specialists by scripts/build_context.py (none is released).

    <context>/patients.csv     case_id, center, fold, label, PATIENT_FEATURES
    <context>/cores.csv        core_id, case_id, center, fold, label, primus, CORE_FEATURES, TARGETING_FEATURES,
                               angle, depth_mm, location, reg_shift_mm
    <context>/hotspots.csv     case_id, fold, sweep hot-spot FEATURES, label, primus
    <context>/guide_paths.npz  per (fold or "all", centre): the mean annotated needle track (the scanner's usual path)
    <context>/calibration.json sweep frame statistics and the patient / core thresholds per fold

For a patient of the cohort, pass its fold: everything then comes from the other folds only. For a new patient, fold=None
uses the whole cohort.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from tools.location import CoreLocator

from .features import CORE_FEATURES, PATIENT_FEATURES, TARGETING_FEATURES
from .tabpfn import TabPFNComposer


def fold_key(fold) -> str:
    return "all" if fold is None else str(int(fold))


class Context:
    def __init__(self, root, fold: int | None = None):
        self.root, self.fold, self.key = Path(root), fold, fold_key(fold)
        keep = lambda d: d if fold is None else d[d.fold != fold]
        read = lambda name: pd.read_csv(self.root / name, dtype={"case_id": str, "core_id": str})
        self.patients, self.cores = keep(read("patients.csv")), keep(read("cores.csv"))
        self.hotspots = keep(read("hotspots.csv")) if (self.root / "hotspots.csv").exists() else None
        cal = json.loads((self.root / "calibration.json").read_text())
        self.sweep_z, self.thresholds = cal["sweep_z"], cal["thresholds"][self.key]
        guide = np.load(self.root / "guide_paths.npz")
        self.guide_paths = {k.split("_", 1)[1]: guide[k] for k in guide.files if k.split("_", 1)[0] == self.key}

    @cached_property
    def locator(self) -> CoreLocator:
        return CoreLocator(self.cores)

    def calibration_for(self, center) -> dict:
        """The centre's usual needle path and the median |roll| of its cores (or of all cores for an unknown centre)."""
        c = self.cores[self.cores.center == center] if center in set(self.cores.center) else self.cores
        return {"guide_path": self.guide_paths.get(str(center)), "absroll_median": float(c.angle.abs().median())}

    def _composer(self, table: pd.DataFrame, keys, name) -> TabPFNComposer:
        t = table.dropna(subset=["label"])
        return TabPFNComposer(
            keys, t.reindex(columns=keys).values.astype(float), t.label.values.astype(int), threshold=self.thresholds.get(name)
        )

    @cached_property
    def patient_composer(self) -> TabPFNComposer:
        return self._composer(self.patients, PATIENT_FEATURES, "patient")

    @cached_property
    def core_composer(self) -> TabPFNComposer:
        return self._composer(self.cores, CORE_FEATURES, "core")

    @cached_property
    def targeting_composer(self) -> TabPFNComposer:
        return self._composer(self.cores, TARGETING_FEATURES, "targeting")

    @cached_property
    def experience(self):
        from tools.sweep import ExperienceScorer

        return ExperienceScorer(self.hotspots) if self.hotspots is not None and (self.hotspots.label >= 0).any() else None
