"""DINOv3 cancer-segmentation network: a DINOv3 ViT-L/16 backbone with a prompt-free SAM-style mask decoder.

Needs the `dinov3` package (github.com/facebookresearch/dinov3) to build the backbone. The decoder building blocks
(MLPBlock to PositionEmbeddingRandom) are adapted from Segment Anything (Meta Platforms, Apache-2.0; see
models/segment_anything/LICENSE).
"""

from __future__ import annotations

import math
from typing import List, Tuple, Type, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Attention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = self._recombine_heads(attn @ v)
        return self.out_proj(out)


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
                for i in range(depth)
            ]
        )
        self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding, image_pe, point_embedding):
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe)

        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("positional_encoding_gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * math.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims, dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PromptFreeMaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_embedding_dim: int,
        transformer_dim: int = 256,
        transformer_depth: int = 2,
        transformer_num_heads: int = 8,
        transformer_mlp_dim: int = 2048,
        num_output_channels: int = 1,
        num_upscale_steps: int = 4,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if num_upscale_steps < 2:
            raise ValueError("num_upscale_steps must be at least 2")

        self.transformer_dim = transformer_dim
        self.num_output_channels = num_output_channels

        self.encoder_proj = nn.Sequential(
            nn.Conv2d(encoder_embedding_dim, transformer_dim, kernel_size=1, bias=False),
            LayerNorm2d(transformer_dim),
        )
        self.pe_layer = PositionEmbeddingRandom(transformer_dim // 2)
        self.mask_tokens = nn.Embedding(num_output_channels, transformer_dim)
        self.transformer = TwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=transformer_num_heads,
            mlp_dim=transformer_mlp_dim,
        )

        self.base_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        extra_layers: List[nn.Module] = []
        dim = transformer_dim // 8
        for _ in range(num_upscale_steps - 2):
            extra_layers += [LayerNorm2d(dim), nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2), activation()]
        self.extra_upscaling = nn.Sequential(*extra_layers) if extra_layers else nn.Identity()

        upscaled_dim = transformer_dim // 8
        self.output_hypernetworks_mlps = nn.ModuleList(
            [_MLP(transformer_dim, transformer_dim, upscaled_dim, num_layers=3) for _ in range(num_output_channels)]
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        B = image_features.shape[0]

        src = self.encoder_proj(image_features)
        _, D, H, W = src.shape

        image_pe = self.pe_layer((H, W)).unsqueeze(0).expand(B, -1, -1, -1)
        tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        hs, src_out = self.transformer(src, image_pe, tokens)

        src_out = src_out.transpose(1, 2).view(B, D, H, W)
        upscaled = self.base_upscaling(src_out)
        upscaled = self.extra_upscaling(upscaled)

        weights = torch.stack(
            [self.output_hypernetworks_mlps[i](hs[:, i, :]) for i in range(self.num_output_channels)],
            dim=1,
        )

        ub, uc, uh, uw = upscaled.shape
        masks = (weights @ upscaled.view(ub, uc, uh * uw)).view(ub, -1, uh, uw)
        return masks


class ViTMaskDecoderSeg(nn.Module):
    def __init__(self, backbone: nn.Module, decoder: PromptFreeMaskDecoder) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
        return self.decoder(features)


class CancerDetectionOutput(TypedDict):
    cancer_logits: torch.Tensor


class BmodeCancerSegmentationAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.ones(1))
        self.register_buffer("bias", torch.zeros(1))

    def forward(self, *, bmode: torch.Tensor, **kwargs) -> CancerDetectionOutput:
        logits = self.model(bmode)
        logits = logits / self.temperature + self.bias
        return {"cancer_logits": logits}


def build_dinov3_head(
    image_size: int = 384,
    backbone_name: str = "dinov3_vitl16",
    freeze_backbone: bool = False,
    transformer_dim: int = 256,
    transformer_depth: int = 2,
    transformer_num_heads: int = 8,
    transformer_mlp_dim: int = 2048,
    num_output_channels: int = 1,
    num_upscale_steps: int = 4,
    pretrained_backbone: bool = True,
) -> BmodeCancerSegmentationAdapter:
    import dinov3.hub.backbones

    backbone = dinov3.hub.backbones.__dict__[backbone_name](pretrained=pretrained_backbone)
    if freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False

    decoder = PromptFreeMaskDecoder(
        encoder_embedding_dim=backbone.embed_dim,
        transformer_dim=transformer_dim,
        transformer_depth=transformer_depth,
        transformer_num_heads=transformer_num_heads,
        transformer_mlp_dim=transformer_mlp_dim,
        num_output_channels=num_output_channels,
        num_upscale_steps=num_upscale_steps,
    )

    inner = ViTMaskDecoderSeg(backbone, decoder)
    return BmodeCancerSegmentationAdapter(inner)
