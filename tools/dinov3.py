"""DINOv3: an independent per-pixel cancer likelihood on one B-mode frame (DINOv3 ViT-L/16 backbone with a mask decoder,
trained on NCT2013; see scripts/train_dinov3_head.py). 384 px input, ImageNet normalisation."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from models.dinov3_head import build_dinov3_head

from .base import load_state_dict, to_map

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv3:
    name = "dinov3"
    modalities = ("frame", "cine", "sweep")

    def __init__(self, checkpoint: str, device: str = "cpu"):
        self.device = device
        self.model = build_dinov3_head(pretrained_backbone=False)
        missing, unexpected = self.model.load_state_dict(load_state_dict(checkpoint), strict=False)
        if missing:
            raise RuntimeError(f"DINOv3 checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
        if unexpected:
            logging.warning(f"DINOv3 checkpoint: {len(unexpected)} unused entries")
        self.model.eval().to(device)

    @torch.no_grad()
    def heatmap(self, frame: np.ndarray) -> np.ndarray:
        """(H, W) uint8 frame -> (H, W) cancer likelihood in [0, 1]."""
        t = torch.from_numpy(frame.astype(np.float32))[None, None].repeat(1, 3, 1, 1)
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        t = F.interpolate(t, size=(384, 384), mode="bilinear", align_corners=False, antialias=True)
        t = ((t - MEAN) / STD).to(self.device)
        return to_map(torch.sigmoid(self.model(bmode=t)["cancer_logits"][0, 0]).float().cpu().numpy(), frame.shape)
