"""Compass: study-level csPCa from the whole rotational sweep, every biopsy core and age / PSA (224 px; 16 sweep frames).

Trained on the sweep and the cores together. It also runs on either alone (a case without cores, or without a sweep), with a
warning, since that was never validated.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from models.compass import build_compass

from .base import load_state_dict
from .prostnfound import AGE_MEAN, AGE_RANGE, PSA_MEAN, PSA_RANGE, normalise

SIZE, SWEEP_FRAMES = 224, 16


def resize(t: torch.Tensor) -> torch.Tensor:
    return F.interpolate(t, size=(SIZE, SIZE), mode="bilinear", align_corners=False, antialias=True)


class Compass:
    name = "compass"
    modalities = ("study", "all", "sweep", "frame")  # study = all = sweep + cores (as trained); sweep alone; frame = cores alone

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = device
        self.model = build_compass()
        self.model.load_state_dict(load_state_dict(checkpoint), strict=True)
        self.model.eval().to(device)

    def inputs(self, case) -> dict:
        """The model's batch for one case: each core's fire frame (min-max scaled) with its roll, 16 evenly spaced sweep
        frames (scaled by 1/255) with their recorded roll, and the age / PSA prompts repeated per core."""
        cores = [c.frame() for c in case.cores]
        biopsy = [
            ((t - t.min()) / (t.max() - t.min())) if t.max() > t.min() else t
            for t in (torch.from_numpy(f.astype(np.float32) / 255.0)[None].repeat(3, 1, 1) for f in cores)
        ]
        biopsy = resize(torch.stack(biopsy)) if biopsy else torch.empty(0, 3, SIZE, SIZE)
        n = len(cores)
        dev = self.device
        sweep, roll = [], []
        if case.sweep is not None:
            k = np.linspace(0, len(case.sweep) - 1, SWEEP_FRAMES).astype(int)
            frames = torch.stack([torch.from_numpy(case.sweep.frame(i).astype(np.float32) / 255.0)[None].repeat(3, 1, 1) for i in k])
            sweep, roll = [resize(frames).to(dev)], [torch.tensor(case.sweep.roll[k], dtype=torch.float32, device=dev)]
        return {
            "biopsy_image": [biopsy.to(dev)],
            "biopsy_roll_angle": [torch.tensor([c.roll_deg for c in case.cores], dtype=torch.float32, device=dev)],
            "sweep": [sweep],
            "sweep_roll_series": [roll],
            "biopsy_age": [torch.full((n, 1), normalise(case.age, AGE_RANGE, AGE_MEAN), device=dev)],
            "biopsy_psa": [torch.full((n, 1), normalise(case.psa, PSA_RANGE, PSA_MEAN), device=dev)],
        }

    @torch.no_grad()
    def predict(self, case) -> dict:
        """{"patient_risk": sigmoid of the csPCa logit, "core_scores": {core_id: csPCa logit - benign logit}}."""
        if case.sweep is None and not case.cores:
            raise ValueError("Compass needs the rotational sweep, the biopsy cores, or both")
        if case.sweep is None or not case.cores:
            alone = "cores" if case.sweep is None else "sweep"
            logging.warning(f"Compass on the {alone} alone: it was trained on the sweep and the cores together")
        out = self.model(**self.inputs(case))
        core = out["biopsy_logits"][0]
        return {
            "patient_risk": float(torch.sigmoid(out["logits"][0, 1])),
            "core_scores": {c.core_id: float(core[i, 1] - core[i, 0]) for i, c in enumerate(case.cores)},
        }
