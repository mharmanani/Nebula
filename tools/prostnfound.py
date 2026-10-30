"""ProstNFound+: per-pixel cancer likelihood on one B-mode frame, prompted with age and PSA (256 px input)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from models.prostnfound import build_prostnfound

from .base import load_state_dict, to_map

# Prompt normalisation used in training (min-max; a missing value takes the training average).
AGE_RANGE, AGE_MEAN = (0.0, 79.0), 62.5816
PSA_RANGE, PSA_MEAN = (0.2, 32.95), 6.821426488456866
PSA_DENSITY_PROMPT = 0.2


def normalise(value, value_range, mean) -> float:
    v = mean if value is None or value != value else float(value)
    return (v - value_range[0]) / (value_range[1] - value_range[0])


class ProstNFound:
    name = "prostnfound"
    modalities = ("frame", "cine", "sweep")

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = device
        self.model = build_prostnfound()
        self.model.load_state_dict(load_state_dict(checkpoint, strip_prefix="model."), strict=True)
        self.model.eval().to(device)

    @torch.no_grad()
    def heatmap(self, frame: np.ndarray, age=None, psa=None) -> np.ndarray:
        """(H, W) uint8 frame -> (H, W) cancer likelihood in [0, 1]. The model was trained on vertically flipped frames (probe
        at the top), so the frame is flipped in and the map flipped back."""
        t = torch.from_numpy(np.flipud(frame).astype(np.float32).copy())[None, None].repeat(1, 3, 1, 1)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        t = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False, antialias=True).to(self.device)
        prompt = lambda v: torch.tensor([[v]], device=self.device)
        out = self.model(
            t,
            age=prompt(normalise(age, AGE_RANGE, AGE_MEAN)),
            psa=prompt(normalise(psa, PSA_RANGE, PSA_MEAN)),
            approx_psa_density=prompt(PSA_DENSITY_PROMPT),
        )
        hm = to_map(torch.sigmoid(out["cancer_logits"][0, 0]).float().cpu().numpy(), frame.shape)
        return np.flipud(hm).copy()
