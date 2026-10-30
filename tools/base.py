"""The tool contract shared by every tool the coordinator can call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

InputKind = Literal["frame", "sweep", "study", "clinical", "outputs"]


@dataclass
class Tool:
    name: str
    description: str
    input_kind: InputKind  # what the tool reads: a frame, the sweep, the whole study, the clinical values, or tool outputs
    run: Callable[..., dict]
    arguments: dict | None = None  # JSON-schema properties of the tool's arguments, if any

    def schema(self) -> dict:
        """Function-calling schema given to the coordinator."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {"type": "object", "properties": self.arguments or {}, "required": []},
            },
        }


def load_state_dict(checkpoint, strip_prefix: str | None = None) -> dict:
    """A state dict from a checkpoint file holding either the state dict or {"model": state dict}; `strip_prefix` is removed
    from the keys that start with it (e.g. "model." for checkpoints saved from a wrapper)."""
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state.get("model_state_dict", state)) if isinstance(state, dict) else state
    if strip_prefix:
        state = {k[len(strip_prefix) :] if k.startswith(strip_prefix) else k: v for k, v in state.items()}
    return state


def to_map(prob: np.ndarray, shape) -> np.ndarray:
    """A probability map resized to `shape` (bilinear, through 8 bits as in the paper's pipeline), values in [0, 1]."""
    from PIL import Image

    img = Image.fromarray((np.clip(prob, 0, 1) * 255).astype(np.uint8)).resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(img, np.float32) / 255.0
