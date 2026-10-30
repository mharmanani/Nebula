"""Compass network: study-level csPCa from the rotational sweep and every biopsy core (github.com/mharmanani/Compass).

A shared, frozen ProstNFound+ backbone embeds each biopsy frame and each sweep frame (its class-decoder token); the tokens get
a type embedding (biopsy / sweep) and a sinusoidal roll-angle embedding and go through a BERT aggregator, which returns the
study logits (from [CLS]) and one logit pair per biopsy token. Trimmed to the configuration used in the paper; module names
are those of the original so its checkpoints load unchanged.
"""

import torch
from torch import nn

from .bert_mil import BertMILAggregator
from .prostnfound import build_prostnfound

PROMPT_KEYS = {"biopsy_psa": "psa", "biopsy_age": "age"}  # data key -> ProstNFound+ prompt


def concat_batch(tensors):
    """Concatenate variable-length tensors; also return each row's batch index."""
    index = torch.cat([torch.full((len(t),), i, dtype=torch.long, device=t.device) for i, t in enumerate(tensors)])
    return torch.cat(tensors, dim=0), index


def split_batch(stacked, index):
    return [stacked[index == i] for i in index.unique()]


class FrozenModule(nn.Module):
    """Runs the wrapped module frozen and in eval mode. (Attribute `module` keeps the original checkpoint keys.)"""

    def __init__(self, module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            p.requires_grad = False

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            self.module.eval()
            return self.module(*args, **kwargs)


class BiopsyEncoder(nn.Module):
    """One ProstNFound+ class-decoder token per biopsy frame, with the clinical prompts."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, bmode_list, **prompts_list):
        bmode, index = concat_batch(bmode_list)
        prompts = {k: concat_batch(v)[0] for k, v in prompts_list.items()}
        hidden = self.model(bmode, **prompts)["class_decoder_hidden_states"][0]
        return split_batch(hidden, index)


class SweepEncoder(nn.Module):
    """One ProstNFound+ class-decoder token per sweep frame (no prompts)."""

    def __init__(self, pnf_model):
        super().__init__()
        self.pnf_model = pnf_model

    def forward(self, sweep_list):
        frames, index = concat_batch(sweep_list)
        return split_batch(self.pnf_model(frames)["class_decoder_hidden_states"][0], index)


class SinusoidalRollEmbedding(nn.Module):
    """Sinusoidal embedding of a continuous roll angle (degrees)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        half = embedding_dim // 2
        self.register_buffer("inv_freq", 1.0 / (10000.0 ** (torch.arange(0, half).float() / half)))

    def forward(self, roll):
        freqs = roll.reshape(-1, 1).float() @ self.inv_freq.unsqueeze(0)
        return torch.cat([freqs.sin(), freqs.cos()], dim=-1)


class Compass(nn.Module):
    BIOPSY, SWEEP = 0, 1

    def __init__(self, pnf_backbone, agg, embedding_dim=256):
        super().__init__()
        self.pnf_backbone = pnf_backbone
        self.biopsy_encoder = BiopsyEncoder(pnf_backbone)
        self.sweep_encoder = SweepEncoder(pnf_backbone)
        self.agg = agg
        self.roll_embedding = SinusoidalRollEmbedding(embedding_dim)
        self.type_embedding = nn.Embedding(2, embedding_dim)
        self.input_proj = nn.Identity()

    def forward(self, *, biopsy_image, biopsy_roll_angle, sweep, sweep_roll_series, **prompts):
        """Batched as lists (one entry per case): biopsy_image (N_i, 3, H, W), biopsy_roll_angle (N_i,), sweep: list of
        (T_j, 3, H, W) per case, sweep_roll_series: list of (T_j,) per case; prompts biopsy_psa / biopsy_age: (N_i, 1).
        Returns {"logits": (B, 2), "biopsy_logits": list of (N_i, 2)}."""
        prompts = {PROMPT_KEYS[k]: v for k, v in prompts.items() if k in PROMPT_KEYS}
        if sum(len(b) for b in biopsy_image):
            biopsy = [self.input_proj(h) for h in self.biopsy_encoder(biopsy_image, **prompts)]
        else:  # sweep only
            dim = self.type_embedding.embedding_dim
            biopsy = [torch.empty(0, dim, device=self.type_embedding.weight.device) for _ in biopsy_image]
        tokens = []
        for i, b in enumerate(biopsy):
            dev = b.device
            b = (
                b
                + self.type_embedding(torch.full((len(b),), self.BIOPSY, device=dev))
                + self.roll_embedding(biopsy_roll_angle[i])
            )
            if len(sweep[i]):
                s = self.input_proj(torch.cat(self.sweep_encoder(sweep[i]), dim=0))
                r = torch.cat(sweep_roll_series[i], dim=0)
                s = s + self.type_embedding(torch.full((len(s),), self.SWEEP, device=dev)) + self.roll_embedding(r)
                b = torch.cat([b, s], dim=0)  # biopsy tokens first, then sweep tokens
            tokens.append(b)
        out = self.agg(tokens)
        return {"logits": out["logits"], "biopsy_logits": [out["token_logits"][i][: len(b)] for i, b in enumerate(biopsy)]}


def build_compass(embedding_dim=256, num_hidden_layers=4, intermediate_size=2048) -> Compass:
    """The configuration of the paper: frozen ProstNFound+ backbone (prompts age and PSA), BERT with 4 layers / 8 heads /
    2048 MLP, no dropout, sinusoidal roll embedding, token states detached for the biopsy head."""
    agg = BertMILAggregator(
        embedding_dim,
        num_classes=2,
        num_token_classes=2,
        detach_token_states_for_classification=True,
        num_attention_heads=8,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    return Compass(FrozenModule(build_prostnfound(prompts=("age", "psa"))), agg, embedding_dim)
