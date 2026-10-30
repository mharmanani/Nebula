"""MicroSegNet: prostate (gland) segmentation on one B-mode frame. Public weights, used zero-shot."""

from __future__ import annotations

import numpy as np
import skimage.transform
import torch
from PIL import Image

from models.microsegnet import build_microsegnet


class MicroSegNet:
    name = "microsegnet"
    modalities = ("frame", "cine", "sweep")

    def __init__(self, checkpoint: str | None = None, device: str = "cpu"):
        """`checkpoint`: the TransUNet state dict; None downloads the public prostate weights."""
        self.device = device
        self.model = build_microsegnet(checkpoint).to(device)

    @torch.no_grad()
    def segment(self, frame: np.ndarray) -> np.ndarray:
        """(H, W) uint8 frame -> (H, W) bool mask."""
        x = skimage.transform.resize(frame / 255.0, (224, 224), anti_aliasing=True)
        x = torch.from_numpy(x)[None, None].float().to(self.device)
        mask = (self.model(x).sigmoid()[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
        return np.asarray(Image.fromarray(mask).resize((frame.shape[1], frame.shape[0]), Image.NEAREST)) > 0
