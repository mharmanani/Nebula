"""NeedleNet: the biopsy needle track on one B-mode frame (MicroSegNet architecture trained on needle masks; see
scripts/train_needlenet.py). The raw prediction is replaced by the rotated rectangle that fits its largest component, since
a biopsy needle is straight."""

from __future__ import annotations

import numpy as np
from skimage.draw import polygon
from skimage.measure import label, regionprops

from .microsegnet import MicroSegNet


def fit_rotated_rectangle(mask: np.ndarray) -> np.ndarray:
    """The rotated rectangle with the centroid, orientation and major / minor axis lengths of the mask's largest component."""
    if not mask.any():
        return mask.astype(bool)
    p = max(regionprops(label(mask)), key=lambda r: r.area)
    cy, cx = p.centroid
    major = np.array([np.cos(p.orientation), np.sin(p.orientation)])  # (dy, dx), skimage convention
    minor = np.array([-np.sin(p.orientation), np.cos(p.orientation)])
    half_l, half_w = p.axis_major_length / 2, max(p.axis_minor_length, 1.0) / 2
    corners = np.array([(cy, cx) + sl * half_l * major + sw * half_w * minor for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))])
    rr, cc = polygon(corners[:, 0], corners[:, 1], shape=mask.shape)
    out = np.zeros(mask.shape, bool)
    out[rr, cc] = True
    return out


class NeedleNet(MicroSegNet):
    name = "needlenet"
    modalities = ("frame", "cine")  # not the sweep: no needle is fired during the sweep

    def __init__(self, checkpoint: str, device: str = "cpu"):
        """`checkpoint`: your NeedleNet weights (scripts/train_needlenet.py); unlike MicroSegNet's, none are public."""
        super().__init__(checkpoint, device)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """(H, W) uint8 frame -> (H, W) bool needle track."""
        return fit_rotated_rectangle(super().segment(frame))
