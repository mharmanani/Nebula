"""MedSAM ViT-B image encoder with adapters."""

from functools import partial

import torch
from torch import nn

from .segment_anything.build_sam import sam_model_registry
from .segment_anything.modeling.image_encoder import Attention, Block, ImageEncoderViT, MLPBlock
from .segment_anything.modeling.mask_decoder import MaskDecoder


def build_medsam_adapter(adapter_dim: int = 256, medsam_checkpoint: str | None = None):
    """MedSAM ViT-B with an adapter in every block; only the adapters are trainable. `medsam_checkpoint`: the public MedSAM
    ViT-B weights (needed only when training from scratch)."""
    model = sam_model_registry["vit_b"](checkpoint=medsam_checkpoint, MaskDecoder=partial(MaskDecoder))
    model.image_encoder = wrap_image_encoder_with_adapter(model.image_encoder, adapter_dim=adapter_dim)
    freeze_non_adapter_layers(model.image_encoder)
    wrap_with_interpolated_pos_embedding_(model)
    return model


class Adapter(nn.Module):
    def __init__(self, feature_dim, adapter_dim, init_scale=1e-3):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(feature_dim, adapter_dim)
        self.up_project = nn.Linear(adapter_dim, feature_dim)
        self.act = nn.GELU()

        # initializations to make it close to identity function
        nn.init.uniform_(self.down_project.weight, -init_scale, init_scale)
        nn.init.uniform_(self.up_project.weight, -init_scale, init_scale)
        nn.init.zeros_(self.down_project.bias)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        return self.up_project(self.act(self.down_project(x))) + x


class AdapterAttn(nn.Module):
    def __init__(self, attn: Attention, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()
        self.attn = attn
        embedding_dim = attn.proj.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.adapter(x)
        return x


class AdapterMLPBlock(nn.Module):
    def __init__(self, mlp: MLPBlock, adapter_dim: int, init_scale: float = 1e-3):
        super().__init__()

        self.mlp = mlp
        embedding_dim = mlp.lin1.in_features

        self.adapter = Adapter(embedding_dim, adapter_dim, init_scale=init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = self.adapter(x)
        return x


def wrap_block_with_adapter(block: Block, adapter_dim: int, init_scale: float = 1e-3):
    block.attn = AdapterAttn(block.attn, adapter_dim, init_scale=init_scale)
    block.mlp = AdapterMLPBlock(block.mlp, adapter_dim, init_scale=init_scale)
    return block


def wrap_image_encoder_with_adapter(image_encoder: ImageEncoderViT, adapter_dim: int, init_scale: float = 1e-3):
    new_blocks = torch.nn.ModuleList()
    for block in image_encoder.blocks:
        new_block = wrap_block_with_adapter(block, adapter_dim, init_scale=init_scale)
        new_blocks.append(new_block)

    image_encoder.blocks = new_blocks

    return image_encoder


def freeze_non_adapter_layers(model: nn.Module):
    for name, param in model.named_parameters():
        if "adapter" not in name.lower():
            param.requires_grad = False

    return model


def interpolate_pos_encoding(x, pos_embed):
    npatch_in_h = x.shape[1]
    npatch_in_w = x.shape[2]

    patch_pos_embed = pos_embed

    npatch_native_h = patch_pos_embed.shape[1]
    npatch_native_w = patch_pos_embed.shape[2]

    if npatch_native_h == npatch_in_h and npatch_native_w == npatch_in_w:
        return pos_embed

    w0 = npatch_in_w
    h0 = npatch_in_h
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    w0, h0 = w0 + 0.1, h0 + 0.1
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.permute(0, 3, 1, 2),
        scale_factor=(h0 / npatch_native_h, w0 / npatch_native_w),
        mode="bicubic",
    )
    assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
    return patch_pos_embed


def forward_return_features(image_encoder: ImageEncoderViT, x, return_hiddens=False):
    # "Return hiddens" feature added

    x = image_encoder.patch_embed(x)
    if image_encoder.pos_embed is not None:
        x = x + interpolate_pos_encoding(x, image_encoder.pos_embed)

    hiddens = []
    for blk in image_encoder.blocks:
        x = blk(x)
        if return_hiddens:
            hiddens.append(x)

    x = image_encoder.neck(x.permute(0, 3, 1, 2))

    return (x, hiddens) if return_hiddens else x


def wrap_with_interpolated_pos_embedding_(sam_model):
    type(sam_model.image_encoder).forward = forward_return_features
