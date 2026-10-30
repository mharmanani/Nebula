"""MicroSegNet (TransUNet, R50-ViT-B/16, 224 px, one output channel): prostate segmentation on micro-ultrasound.

Network code from MicroSegNet (Jiang et al., 2024). The public prostate weights are on the Hugging Face Hub
(pfrwilson/micro_seg_net); NeedleNet uses the same network trained on needle masks.
"""

import torch

from .vit_seg_modeling import CONFIGS, VisionTransformer

PUBLIC_WEIGHTS = ("pfrwilson/micro_seg_net", "MicroSegNet.pth")


class MicroSegNetwork(torch.nn.Module):
    """VisionTransformer (TransUNet) returning the segmentation logits only. (Attribute `net` holds the original keys.)"""

    def __init__(self, img_size: int = 224):
        super().__init__()
        conf = CONFIGS["R50-ViT-B_16"]
        conf.n_classes, conf.n_skip = 1, 3
        conf.patches.size = (16, 16)
        conf.patches.grid = (img_size // 16, img_size // 16)
        self.net = VisionTransformer(conf, img_size=img_size, num_classes=1)

    def forward(self, x):
        return self.net(x)[0]


def build_microsegnet(checkpoint: str | None = None) -> MicroSegNetwork:
    """`checkpoint`: a state dict of the TransUNet (the keys of `MicroSegNetwork.net`); None downloads the public prostate
    weights."""
    model = MicroSegNetwork()
    if checkpoint is None:
        from huggingface_hub import hf_hub_download
        checkpoint = hf_hub_download(*PUBLIC_WEIGHTS)
    model.net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.eval()
